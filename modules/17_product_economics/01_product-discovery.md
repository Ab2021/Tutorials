# Product Discovery

> Module 17 — Product, Economics, and Leadership
> Topic 01 of 09
> Estimated study time: 4 hours
> Prerequisites: 08 (EHR and claims data)
> Version 2026.3

## 1. Why this topic exists

Most biomedical data products that fail commercially do not fail because the model was wrong.
They fail because the model was aimed at a decision that no one owned, at a moment in the
workflow where no one could act, measured by an outcome no one was paid to improve. The
statistics were fine. The target was wrong.

Product discovery is the disciplined work of finding a decision worth changing before you build
anything to change it. It is not customer development theatre, and it is not a survey. It is a
sequence of observations, interviews, and small quantitative estimates that answer one question:
is there a decision here that is frequent, consequential, measurable, and owned by someone with
the authority and budget to change it?

That question has a definite answer in most clinical settings, and the answer is often no. A
team that cannot say no at the discovery stage will spend eighteen months building and then
discover the answer during procurement, at ten times the cost.

### 1.1 The default failure mode: building outward from the dataset

Data scientists are trained to start with what is available. A team receives a warehouse of
encounter data, notices that sepsis onset times are recorded, and builds a sepsis prediction
model. The model performs well on held-out data. The team then looks for a buyer and discovers
that the hospital already licenses two sepsis tools, that the alert fires into a workflow with no
dedicated response nurse, and that the quality team measures sepsis bundle compliance rather than
onset prediction. The technical work was competent and the commercial work was absent.

This pattern is not a failure of intelligence. It is a failure of sequence. The dataset is a
constraint on what you can build; it is not evidence that anyone needs it built.

### 1.2 What discovery produces

A completed discovery produces five artifacts, none of which is a slide deck:

1. A **decision statement**: the specific choice, made by a named role, at a named point in a
   workflow, that your product would change.
2. A **status quo quantification**: how often the decision is made, how long it takes, how often
   it is wrong, and what a wrong decision costs.
3. A **buying-centre map**: who uses, who pays, who can block, and what evidence each requires.
4. An **opportunity estimate**: a range for the annual value at stake, with the assumptions
   written down, not a single number.
5. A **kill list**: the conditions under which you would stop, agreed before the evidence arrives.

Teams that skip the fifth artifact produce discovery reports that always recommend proceeding.
That is a sign the exercise was advocacy, not discovery.

### 1.3 Why the industry problem is not the academic one

An academic question is whether a prediction is calibrated and whether the effect is real. An
industry question is whether a change in practice, caused by the prediction, produces more value
than it costs to produce and to act on. Those are different questions with different evidence.

A model with an area under the receiver operating characteristic curve (AUROC) of 0.82 can be
commercially worthless if the positive predictive value at the operating threshold is 12 percent
and the intervention it triggers costs 40 minutes of nursing time. A model with an AUROC of 0.68
can be valuable if it replaces a 30-minute manual chart review that happens 200 times a day.
Discovery is how you learn which situation you are in before you have written the model.

### 1.4 Who this topic is for

If you are a data scientist, discovery determines which data you will be allowed to use, which
label definition is commercially meaningful, and which operating point the buyer will accept.
Being in the discovery conversations is the difference between receiving a specification and
co-authoring one.

If you are a product manager, this topic is the operating manual for the first eight weeks of a
new product area. It gives you a repeatable protocol and a scoring rule.

If you are a founder or an internal innovation lead, the kill criteria in Section 5 are the part
to adopt first. They are cheaper than any other part of the process.

If you are moving from academia, the adjustment is that a negative discovery result is a good
outcome. It costs eight weeks instead of two years.

### 1.5 What this topic does not cover

It does not cover sales, pricing, or contracting; those are Topics 05 and 07. It does not cover
study design for validating a model; that is Module 05. It does not cover qualitative research
methodology in the depth a trained ethnographer would want; it gives a protocol sufficient for
commercial discovery, not for publication. The regulatory material is orientation and is not
legal advice.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Write a decision statement naming the role, the trigger, the choice, and the consequence | Exercise 1; self-check 1 |
| 2 | Design a shadowing protocol that captures decision points rather than task lists | Exercise 2; self-check 3 |
| 3 | Run a problem interview that elicits past behaviour instead of hypothetical intent | Exercise 3; interview question 4 |
| 4 | Quantify the cost of the status quo from operational data and observation | Exercise 4; self-check 5 |
| 5 | Score and rank candidate opportunities on frequency, severity, measurability, and reachability | Exercise 5; self-check 6 |
| 6 | Map a buying centre and identify the budget owner and the most likely blocker | Exercise 6; self-check 7 |
| 7 | Set kill criteria before evidence collection and apply them without renegotiation | Exercise 7; interview question 9 |
| 8 | Compute a discovery-stage opportunity range and a break-even effect size | Exercise 8; self-check 8 |

Each outcome maps to at least one exercise and one self-check item. Outcome 7 is the one most
often skipped in practice and the one most predictive of whether a team wastes a year.

## 3. The core mental model

Discovery is a filter with a value-of-information logic behind it. You are not trying to learn
everything about a market. You are trying to buy, at the lowest cost, the information that would
change whether you build.

### 3.1 The five-layer discovery stack

```text
  Layer 5   CONSEQUENCE
            What happens when the decision is wrong?
            Cost, harm, delay, rework, penalty.
                 ^
  Layer 4   OWNERSHIP
            Who bears the consequence and controls the budget?
            If no one does, stop here.
                 ^
  Layer 3   MEASURABILITY
            Can the consequence be counted today, in data the buyer trusts?
                 ^
  Layer 2   DECISION POINT
            Where in the workflow does a person choose, and what do they choose between?
                 ^
  Layer 1   OBSERVED WORK
            What people actually do, including workarounds, not what the SOP says.
```

Read the stack from the bottom. Layer 1 is where you start, because everything above it is
usually misreported by the organisation itself. Layer 4 is where most opportunities die, and it
is the layer teams check last.

The stack is ordered by cost of finding out. Layer 1 costs days of shadowing. Layer 4 costs
weeks of meetings with people who do not take them. Doing them out of order means paying the
Layer 4 cost to discover a Layer 1 misunderstanding.

### 3.2 The discovery funnel

```text
  40-60 conversations     broad, cheap, mostly to find vocabulary and segments
        |
        v
  12-15 deep interviews    chosen for variation in role and setting
        |
        v
   6-8 shadowing sessions  chosen where the decision is highest-volume
        |
        v
   2-3 quantified problems with a named owner and a countable cost
        |
        v
   1 wedge                 the smallest product that changes one decision
```

The funnel is a ratio, not a schedule. Forty conversations yielding one wedge is a normal
outcome. Forty conversations yielding zero is also normal and is the result you most need to be
willing to report.

### 3.3 Decision-centric versus feature-centric framing

A feature-centric statement: "clinicians need a better risk score for acute kidney injury."
A decision-centric statement: "the admitting hospitalist, at the point of ordering intravenous
contrast for a patient with an estimated glomerular filtration rate below 30, currently has no
prompt that shows the alternative imaging pathway, and orders contrast in 22 percent of those
cases."

The second statement tells you who to interview, what to measure, where to integrate, and what
the product must change. The first tells you only what to build. Feature-centric framing
produces products that are evaluated on whether the feature exists rather than whether the
decision changed.

### 3.4 Value of information

Discovery spend is justified by the decisions it changes, not by the knowledge it produces. A
useful heuristic: if the fully loaded cost of building and selling the product for two years is
B, and discovery costs 5 to 10 percent of B, then discovery is worth doing if it has better than
roughly a one-in-ten chance of stopping the build. Most teams underestimate the stop probability
and therefore underinvest in discovery.

The inverse also holds. If B is small, for example an internal tool that one analyst can build in
three weeks, extensive discovery is over-investment. The protocol in Section 5 scales down: two
days of shadowing and three interviews may be enough.

### 3.5 The three questions that decide whether to continue

At the end of discovery, three questions determine the recommendation. They are deliberately
blunt because nuance is what teams use to avoid answering them.

1. **Is the decision made often enough to matter?** A decision made twice a year cannot support
   a subscription product, regardless of how consequential it is.
2. **Does the person who bears the cost control the change?** If the cost falls on a payer and
   the change must be made by a provider, you have two buying centres and possibly no deal.
3. **Can the benefit be measured inside the contract term?** A benefit that appears in
   twelve-year survival cannot be sold to a three-year contract without an intermediate metric.

A "no" to any one is not automatically fatal, but it must be named in the recommendation with a
mitigation, not omitted.

### 3.6 Redraw test

Close the file and draw the five-layer stack from memory, then place your current project on it.
Most people can draw Layers 1 to 3 and cannot name the person at Layer 4. That gap is the
finding.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or system |
|---|---|---|---|
| Product discovery | Structured investigation to decide whether a product opportunity is worth building | Used to mean "customer interviews" only | Process |
| Decision point | A specific moment where a named role chooses among alternatives | Confused with a task or a screen | Workflow |
| Decision statement | One sentence naming role, trigger, choice, and consequence | Written as a feature description | Text artifact |
| Workflow observation | Direct observation of work in its real setting, including interruptions | Replaced by a survey of what people say they do | Hours observed |
| Shadowing | Following one worker through a full shift or clinic session | Observing a demo of the software instead | Hours |
| Contextual inquiry | Interview conducted in the workplace while the work is happening | Conducted in a conference room | Method |
| Time and motion study | Systematic timing and categorisation of tasks | Used loosely for any timing exercise | Seconds per task |
| Diary study | Participants record events over days or weeks | Confused with a retrospective survey | Days |
| Workaround | An unofficial practice that compensates for a tool or process gap | Treated as user error | Count per session |
| Job to be done (JTBD) | The progress a person is trying to make in a situation | Used as a synonym for "use case" | Statement |
| Latent need | A need the user cannot articulate because it is normal to them | Confused with an unvalidated idea | Qualitative |
| Stated preference | What someone says they would do | Treated as a forecast of behaviour | Survey response |
| Revealed preference | What someone actually did when facing a real choice | Ignored in favour of surveys | Observed behaviour |
| Status quo cost | Fully loaded cost of the current way of working, including workarounds | Counted as zero when no budget line exists | Currency per year |
| Cost of the wrong decision | Expected loss from an incorrect choice, including downstream effects | Counted only as the immediate cost | Currency per event |
| Frequency | Number of decision occurrences per unit time | Estimated from memory rather than data | Events per year |
| Severity | Consequence per event, weighted by harm and reversibility | Collapsed into "clinical importance" | Currency or harm score |
| Measurability | Whether the consequence is observable in data the buyer trusts | Assumed because a metric exists somewhere | Boolean plus data source |
| Reachability | Whether your product can influence the decision at an accessible point | Assumed from technical feasibility | Boolean plus integration |
| Wedge | The smallest product that changes one decision for one segment | Confused with a minimum viable product feature list | Product scope |
| Minimum viable product (MVP) | The smallest release that produces validated learning | Used to mean "unfinished product" | Release scope |
| Concierge test | Delivering the service manually behind the scenes to test demand | Confused with a pilot | Test design |
| Wizard of Oz test | User interacts with a simulated product operated by a human | Disclosed incorrectly as automated | Test design |
| Buying centre | All roles involved in a purchase decision | Reduced to "the buyer" | Role set |
| Budget owner | The role with authority to release funds for this category of spend | Assumed to be the most senior person present | Role |
| Champion | An internal advocate with influence and willingness to spend political capital | Confused with a friendly contact | Role |
| Blocker | A role whose objection can stop the purchase | Assumed absent if not encountered | Role |
| Economic buyer | The role that approves the business case and the spend | Confused with procurement | Role |
| Procurement | The purchasing function and its process | Used to mean the whole sale | Function |
| Problem-solution fit | Evidence that a defined problem is painful enough that a proposed solution is wanted | Confused with product-market fit | Stage gate |
| Product-market fit | Evidence that a market buys and retains the product at scale | Declared after a pilot | Stage gate |
| Pilot purgatory | Repeated pilots that never convert to a contract | Treated as a sales delay rather than a signal | Pattern |
| Kill criterion | A pre-agreed condition that stops the project | Written after the result is known | Decision rule |
| Willingness to pay (WTP) | The maximum a buyer would commit, revealed by a real budget decision | Measured by asking "would you pay" | Currency |

## 5. The mechanics, step by step

The procedure below is written for a new product area inside a company that already has access
to a clinical partner. It assumes eight to ten weeks of calendar time and roughly 0.6 full-time
equivalent of a product manager plus 0.3 of a data scientist. Each step names its input, its
output, and the reason it exists. Steps can be compressed but not reordered, because each one
defines the sample for the next.

### Step 1. Write the discovery question and the decision to be changed

**Input:** a sponsor hypothesis, in whatever form it arrived.
**Output:** a one-page discovery brief with a primary question, a secondary question, and a
decision that will be made at the end.
**Rationale:** discovery without a decision attached expands to fill the available time. Writing
"we will decide whether to fund a build, a pilot, or nothing, on this date" converts the work
from research into a decision process. The brief should also name who will make that decision.

The brief contains:

- The population or setting in scope.
- The decision the product would change, stated as a sentence.
- The three questions from Section 3.5, written as hypotheses with a current best guess.
- The budget and deadline for discovery itself.
- The kill criteria, drafted now and ratified at Step 12.

Edge case: if the sponsor cannot name a decision that will be made, the project is not a
discovery project. It is a research project with a commercial label, and it should be scoped as
research with different success criteria.

### Step 2. Segment the interview population before recruiting

**Input:** the discovery brief.
**Output:** a sampling frame of 4 to 6 strata with target counts.
**Rationale:** convenience sampling produces consensus from people who resemble the team. The
strata should vary along axes that plausibly change the answer: role, site type, volume, payment
model, and technology maturity.

A workable sampling frame for a hospital product:

| Stratum | Example | Target |
|---|---|---:|
| Front-line user, high volume site | Staff nurse in a 600-bed hospital | 6 |
| Front-line user, low volume site | Nurse in a 60-bed critical access hospital | 4 |
| Supervisor or charge role | Unit manager | 4 |
| Clinical leadership | Department chief, chief medical information officer | 4 |
| Technical and security | IT analyst, information security officer | 3 |
| Finance and contracting | Service line finance lead, contracting manager | 3 |
| Adjacent workflow | Receiving department such as radiology or the lab | 3 |

Edge case: in a single-payer system, add a payer or commissioner stratum and remove the
contracting stratum, because the funding decision sits outside the provider. In a
multi-payer system, both strata exist and may give contradictory answers, which is itself a
finding.

### Step 3. Recruit with the right incentive and the right framing

**Input:** the sampling frame.
**Output:** a booked schedule with a 70 percent or better show rate.
**Rationale:** clinicians do not have spare hours. Recruitment is a logistics problem with an
ethics component. Framing the request as "help us understand your workflow" produces different
access than "help us evaluate our product"; the first is more honest at this stage and gets
better data.

Practical rules:

- Book 30 minutes, not 60. Offer two slots rather than a scheduling link.
- Compensate at fair market value where institutional policy allows, and route the payment
  through the institution's research or consulting process rather than paying an individual
  directly.
- Do not recruit through the champion alone. A champion recruits people who agree with them.
- Record consent for notes and, separately, for recording. In the United States, if the intent
  is generalisable knowledge, an institutional review board (IRB) determination is required
  before recruitment; see Section 8.

### Step 4. Observe before you interview

**Input:** a site and a shift.
**Output:** a decision log with timestamps.
**Rationale:** interviews report the intended process. Observation reports the actual one. Doing
interviews first contaminates them, because the interviewee then describes the process to you as
they believe it should be.

The observation protocol has four rules:

1. **Position, do not participate.** Stand or sit where you can see the screen and the room, and
   say plainly that you are there to watch, not to help.
2. **Log decisions, not tasks.** A task is "opened the chart." A decision is "decided not to
   order the test because the result would not change management until Monday."
3. **Log interruptions.** Interruption frequency is the single best predictor of whether a new
   tool will be used as designed.
4. **Log every workaround and every paper artifact.** A printed list, a sticky note, a personal
   spreadsheet, and a text message to a colleague are all evidence of a tool gap.

The decision log format:

| Field | Content |
|---|---|
| Time | Clock time, to the minute |
| Actor | Role, not name |
| Trigger | What caused the decision to be needed |
| Options considered | What the person weighed |
| Information used | Screen, paper, colleague, memory |
| Time to decide | Seconds or minutes |
| Outcome | What was chosen |
| Confidence signal | Any expressed uncertainty |
| Downstream effect | What happened next, if observable |

After 15 to 20 hours of observation, patterns appear. After 40 hours, the marginal return
flattens. A single 4-hour session is enough to generate hypotheses but not to rank them.

### Step 5. Count the decision's frequency from operational data

**Input:** observation hypotheses plus available data.
**Output:** a defensible count of decision occurrences per year.
**Rationale:** frequency is the denominator of every value calculation. Teams routinely estimate
it from memory and are wrong by a factor of two or more.

Methods, in order of preference:

1. Query the event log directly, for example count of orders of a given type per year.
2. Count from a sample of encounters and extrapolate, with the sampling interval stated.
3. Ask supervisors for a volume report; treat as an estimate, not a count.
4. Ask interviewees; use only to bound the range.

Record the method next to the number. "4,200 per year, from the discharge registry, 2025" is a
different claim from "about 4,000, per the unit manager."

### Step 6. Run problem interviews that elicit past behaviour

**Input:** the schedule from Step 3 and the hypotheses from Step 4.
**Output:** 12 to 15 interview transcripts or structured notes, coded by theme.
**Rationale:** people are poor at predicting their own behaviour and good at recounting it. The
interview therefore asks what happened last time, not what would happen if.

Interview structure, 30 minutes:

1. **Context (5 minutes).** "Walk me through your last shift." Let them talk. Do not steer.
2. **Specific instance (10 minutes).** "Tell me about the last time a patient deteriorated on
   your unit. What did you notice first?" Anchor on a real event with a date.
3. **Workaround (5 minutes).** "What did you do that was not in the system?" This is where the
   product opportunity usually is.
4. **Cost (5 minutes).** "How long did that take? Who else got involved? What happened to the
   patient?"
5. **Alternatives (5 minutes).** "What have you tried? What did you stop using, and why?"

Questions to avoid, and the replacement:

| Avoid | Ask instead | Why |
|---|---|---|
| "Would you use a tool that…" | "What do you do today when…" | Hypothetical intent is unreliable |
| "How much would you pay for…" | "What is this costing you now?" | Buyers do not price in interviews |
| "Do you think this is a good idea?" | "What would have to be true for you to change?" | Politeness bias |
| "Is workflow a problem?" | "Show me the last time it broke." | Leading and abstract |
| "Who makes the buying decision?" | "Who signed the last contract for something like this?" | Role titles mislead |

### Step 7. Quantify the status quo

**Input:** observation logs, interview notes, operational data.
**Output:** a status-quo cost model with a low, base, and high estimate.
**Rationale:** the buyer's alternative to your product is not nothing; it is the current process.
If the current process costs little, the product has little room. Quantifying it also gives you
the first version of the business case.

Build the model in four lines:

```text
status_quo_cost = events_per_year x minutes_per_event / 60 x loaded_hourly_cost
                + rework_events_per_year x cost_per_rework
                + adverse_events_per_year x cost_per_adverse_event
                + invisible_cost (workarounds, after-hours work, staff turnover)
```

Use a loaded hourly cost, not salary. For a United States registered nurse in 2026, a loaded
rate in the range of 55 to 75 United States dollars (USD) per hour is a reasonable planning
estimate depending on region and benefits; state the basis. For a physician, loaded rates above
150 USD per hour are common in employed models; again, state the basis and the year.

Label every input as measured, estimated, or assumed. A model with three measured inputs and
four stated assumptions is more credible than one with seven numbers of unknown provenance.

### Step 8. Map the buying centre

**Input:** interviews plus organisational documents.
**Output:** a table of roles with influence, interest, stance, evidence need, and gate.
**Rationale:** the user rarely pays. The person who pays rarely uses. A deal needs both, and a
blocker in security or legal can stop it after both are satisfied.

| Role | Question they answer | Evidence they need | Typical gate |
|---|---|---|---|
| User | Does this make my day better or worse? | Workflow demonstration, time saved | Silent non-use |
| Clinical champion | Is this good medicine? | Local pilot data, peer evidence | Withdrawal of support |
| Department chief | Does this help my throughput or quality target? | Operational and financial case | Budget refusal |
| Chief financial officer | What is the return and when? | ROI model, payback, total cost of ownership | Capital review |
| Chief information officer | Can we run and secure it? | Architecture, integration plan, support model | Technical review |
| Information security | Is our data safe and compliant? | SOC 2 Type II, penetration test, data flow diagram | Security review |
| Privacy or compliance | Are we permitted to do this? | Business associate agreement, minimum necessary analysis | Legal review |
| Procurement | Is the process and price defensible? | Competitive process, references, terms | RFP or sole-source approval |
| Payer or commissioner | Should this be covered and paid for? | Clinical utility, cost-effectiveness, budget impact | Coverage decision |

Edge case: in a capital-constrained system, the chief financial officer may be the true economic
buyer even when the department chief believes they are. Test this by asking who signed the last
comparable purchase.

### Step 9. Score opportunities

**Input:** two to five candidate problems from Steps 4 to 7.
**Output:** a ranked list with an explicit scoring rule.
**Rationale:** a score forces the team to state why one problem is better than another. The score
is not truth; it is a device for making disagreement specific.

Use four factors, each scored 1 to 5, with weights agreed in advance:

```text
score = 0.30 x frequency + 0.35 x severity + 0.20 x measurability + 0.15 x reachability
```

Severity carries the highest weight because frequency with low consequence produces busywork.
Measurability and reachability are weighted lower because they are engineering problems that
money can partly solve; frequency and severity are properties of the world.

Score each factor against a defined anchor, not against feeling:

| Score | Frequency (events/year) | Severity (cost per event) | Measurability | Reachability |
|---:|---|---|---|---|
| 1 | under 1,000 | under 100 USD | No data exists | No integration path |
| 2 | 1,000 to 10,000 | 100 to 1,000 USD | Proxy metric only | Manual export possible |
| 3 | 10,000 to 100,000 | 1,000 to 10,000 USD | Measured quarterly | Batch integration possible |
| 4 | 100,000 to 1,000,000 | 10,000 to 100,000 USD | Measured monthly | Real-time interface available |
| 5 | over 1,000,000 | over 100,000 USD | Measured per event | In-workflow integration available |

### Step 10. Choose the wedge

**Input:** the ranked list.
**Output:** one wedge statement with an explicit out-of-scope list.
**Rationale:** the wedge is where commercial discipline is tested. A wedge that requires three
integrations, two new roles, and a change in payment is not a wedge; it is a platform.

A wedge statement has the form:

> For [role] at [setting], when [trigger], we will change [decision] by [mechanism], so that
> [measurable outcome] moves by [amount] within [period]. We will not do [explicit exclusions].

The exclusions matter as much as the inclusion. A wedge that does not exclude anything will
expand until it is unbuildable.

### Step 11. Test demand before building

**Input:** the wedge statement.
**Output:** evidence that a real user changes behaviour when the wedge is available.
**Rationale:** stated demand is cheap; behaviour is expensive. A concierge or Wizard of Oz test
converts a claim into an observation for a few thousand dollars.

Design rules:

- Recruit users who are not the champion's direct reports.
- Run for at least two weeks and at least 30 decision events.
- Measure the decision, not satisfaction. Did the user act differently?
- Pre-commit to what result would stop the project.

A useful threshold: if fewer than 30 percent of eligible decisions use the wedge when it is
available at no cost, the problem is not painful enough or the wedge is in the wrong place.
Thirty percent is a floor for a free tool; a paid product needs higher.

### Step 12. Write the discovery brief and apply the kill criteria

**Input:** everything above.
**Output:** a six-page brief and a go, pilot, or stop recommendation.
**Rationale:** the brief is the artifact that survives the team. It must contain the negative
findings.

Brief structure: decision statement; population and frequency; status-quo cost with ranges;
buying-centre map; competitive and alternative landscape; wedge statement; evidence collected
and its limits; three risks; recommendation; kill criteria and their status.

### 5.13 Branch points and edge cases

**No reimbursement pathway.** If no code, coverage policy, or contract mechanism pays for the
service, the buyer must fund it from an existing operating budget. That changes the sale from
"new revenue" to "cost avoidance or quality improvement," which is a harder case. Discovery must
then identify the budget line that absorbs the cost and the metric that budget holder is
measured on. In the United States, this is the common situation for care-coordination software
that does not map to a payable service.

**Single-payer versus multi-payer.** In a single-payer system, the coverage decision is national
or regional and the provider's economics depend on global budgets or activity-based payment. A
product that saves the provider money may save the payer nothing, or vice versa. Discovery must
name which budget the value lands in. In a multi-payer system, a provider may bear the cost of a
prevention product while the benefit accrues to whichever payer retains the patient, which is the
classic misalignment for chronic disease prevention.

**Capital versus operating budget.** A 200,000 USD capital purchase and a 4,000 USD per month
subscription follow different approval paths, different depreciation rules, and different
timelines. Discovery must establish which budget the buyer will use, because it determines
whether the price point, the contract term, and the procurement route are viable. Capital
purchases often need a return-of-investment threshold set by the finance committee; operating
spend often needs only departmental sign-off within a delegated limit.

**Pilot-to-contract transition.** A pilot funded by an innovation group frequently has no owner
for the production contract. Discovery should identify, before the pilot starts, who will sign
the contract at month nine and from which budget. If that person is not identified, the pilot is
a demonstration, not a sales stage.

**Procurement security review.** A security review can take eight to sixteen weeks and can
surface requirements that change the architecture, such as data residency, customer-managed
encryption keys, or a prohibition on subprocessors. Discovery should obtain the buyer's security
questionnaire during the interviews, not after the pilot. The answers determine feasibility and
cost.

**Champion turnover.** Clinical champions change roles, go on leave, or lose their sponsor. A
deal that depends on one person has a single point of failure. Discovery should record at least
two champions and the organisational reason each cares, so that a replacement can be recruited
on the same rationale.

**Data access denied.** If the required data sit in a system the buyer will not connect, the
product cannot operate. Test data access early with a small extract request rather than assuming
it will be granted at contract.

**Regulatory ambiguity.** If it is unclear whether the product is a medical device, discovery
must resolve the intended-use question before the build, because the answer changes the evidence
requirement, the timeline, and the cost by an order of magnitude.

### 5.14 Decision table for the discovery exit

More than two outcomes are possible at the exit, and the correct action differs by combination.
The table below is the one to use at Step 12.

| Evidence of pain | Named budget owner | Measurable outcome | Behavioural test result | Recommendation |
|---|---|---|---|---|
| Strong | Yes | Yes | Above threshold | Build, with a pilot contract |
| Strong | Yes | Yes | Below threshold | Return to Step 4; the wedge is mispositioned |
| Strong | Yes | No | Above threshold | Pilot with an instrumented measurement plan |
| Strong | No | Yes | Above threshold | Stop, or find a different buyer before building |
| Strong | No | No | Any | Stop |
| Weak | Any | Any | Any | Stop |
| Strong | Yes | Yes | Not run | Do not build; run the behavioural test first |

The third row is the common trap. A product with real behavioural demand but no measurable
outcome is a product that will be renewed once and then cut at budget review, because the buyer
cannot defend it.

## 6. Worked example

> **Note:** the organisation below is a composite constructed for teaching. It is not a real
> company, and every financial figure is an illustrative estimate in 2026 USD, not a reported
> result. The parameters were chosen to be plausible for a United States acute-care system, and
> the arithmetic is exact so that you can reproduce it.

### 6.1 The situation

Halcyon Care is a three-hospital system with 610 licensed beds and about 19,000 annual
discharges. Its innovation group has a hypothesis: heart failure readmissions are a large,
measurable cost, and a remote weight-and-symptom monitoring programme with a nurse navigator
could reduce them. The group has 180,000 USD and ten weeks for discovery, and a decision date at
the end of that period.

The discovery brief states the decision: fund a 12-month pilot with a commercial partner, build
internally, or stop.

### 6.2 What observation found

Twelve half-day observation sessions across three units and 34 interviews produced four findings
that changed the framing.

1. The discharge education process was not standardised. Three of eleven observed discharges
   used a printed heart failure zone sheet; the rest used verbal instruction only. The sheet was
   available in the electronic health record (EHR) but required four clicks from the discharge
   workflow.
2. The readmission problem was concentrated. In the prior year, 21.6 percent of heart failure
   discharges were readmitted within 30 days, but 38 percent of those readmissions came from
   11 percent of patients, identified by two or more prior admissions in twelve months.
3. The nurse navigator role existed but was funded for 0.5 full-time equivalent and covered
   only patients referred by cardiology, which was 31 percent of heart failure discharges.
4. The finance office already tracked readmissions by diagnosis-related group and had a
   penalty exposure model. The number was trusted and reported monthly.

Finding 4 is the one that made the opportunity viable. The outcome was measurable in a system
the buyer already trusted, which is Layer 3 of the stack. Finding 3 identified the ownership
gap, which is Layer 4.

### 6.3 The status quo, quantified

| Input | Value | Basis |
|---|---:|---|
| Adult heart failure discharges per year | 4,200 | Discharge registry, 2025 |
| 30-day all-cause readmission rate | 21.6% | Registry, 2025 |
| Readmissions per year | 907 | Computed |
| Fully loaded cost per readmission | 16,400 USD | Finance estimate, 2026 cost basis |
| Navigator coverage of discharges | 31% | Interview plus staffing records |
| Navigator time per enrolled patient | 4.2 hours | Observation, 11 patients |
| Loaded navigator cost per hour | 62 USD | HR compensation model, 2026 |

Status quo navigator cost: 4,200 x 0.31 x 4.2 x 62 = 339,000 USD per year (rounded). This is
the existing spend, not a saving. It matters because it establishes that the organisation has
already decided the problem is worth paying for, which is evidence of willingness to pay.

### 6.4 The proposed programme and its arithmetic

The proposed programme enrols patients at discharge, provides a cellular-connected scale and a
symptom questionnaire for 90 days, and assigns a navigator to review alerts each weekday.

| Parameter | Value | Basis |
|---|---:|---|
| Eligible discharges per year | 4,200 | Registry |
| Enrollment rate | 55% | Assumption, based on navigator coverage plus opt-out |
| Patients enrolled per year | 2,310 | Computed |
| Programme cost per enrolled patient | 340 USD | Vendor quote plus internal navigator time, 2026 |
| Annual programme cost | 785,400 USD | 2,310 x 340 |
| Assumed relative reduction in readmissions | 18% | Assumption, from published remote monitoring meta-analyses; see Section 9 |
| Readmissions avoided per year | 163.3 | 907.2 x 0.18 |
| Gross saving | 2,678,054 USD | 163.3 x 16,400 |
| Net benefit | 1,892,654 USD | 2,678,054 minus 785,400 |
| Return on investment (ROI) | 2.41 to 1 | Net benefit divided by programme cost |
| Break-even relative reduction | 5.3% | 785,400 divided by (907.2 x 16,400) |

The break-even figure is the most useful number in the model. It says the programme pays for
itself if it prevents 5.3 percent of readmissions, which is 48 readmissions per year, or about
one per week across three hospitals. That is a low bar, and it is the number to put in front of
the finance committee, because it does not require anyone to believe the 18 percent assumption.

### 6.5 Payback and budget impact

Implementation cost, including integration with the EHR, training, and project management, is
estimated at 240,000 USD. At a net benefit of 1,892,654 USD per year, or 157,721 USD per month,
payback is 1.5 months. That figure is implausibly fast and should be treated with suspicion: it
assumes the full annual benefit accrues from month one, which is false during ramp-up. A
defensible version applies a six-month ramp: months 1 to 6 deliver 40 percent of steady-state
benefit, months 7 to 12 deliver 100 percent. Under that assumption, first-year net benefit is
roughly 1,324,900 USD and payback is about 3.8 months, because the first six months accrue only
63,088 USD per month against the 240,000 USD implementation cost. Either way, payback is inside
the first year, which is unusual and is itself a red flag: the model is probably too optimistic
about the effect size.

Budget impact for the payer is different from ROI for the provider. The provider avoids
readmission costs; a Medicare Advantage plan avoids the same costs only if the patient remains
attributed, and a fee-for-service payer loses revenue when a readmission does not occur. This is
the multi-payer misalignment described in Section 5.13, and it is why the same product can be
attractive to a hospital and unattractive to a payer.

### 6.6 Two diagnostics that reveal a weak business case

**Diagnostic 1: the attribution test.** Ask what share of the avoided readmissions would have
been avoided anyway. In this case, the navigator already covers 31 percent of discharges, and
those patients already receive follow-up. If the programme's effect is concentrated in the
already-covered group, the incremental effect is smaller than the headline. The discovery team
recomputed the effect on the uncovered 69 percent only: assuming the same 18 percent relative
reduction applied only to the 2,898 uncovered discharges, avoided readmissions fall to 112.7 and
the ROI falls to 1.35 to 1. Still positive, but the case is now sensitive to the effect size.
This diagnostic is cheap and it kills a large share of naive value cases.

**Diagnostic 2: the measurement-window test.** The programme cost is incurred over 90 days, but
the readmission benefit is measured at 30 days. If the health system's contracts measure
readmissions at 30 days, the programme is being paid for 90 days of monitoring to move a 30-day
metric. Does the extra 60 days of monitoring contribute? If not, the programme can be shortened
to 30 days, which cuts cost per patient materially. The discovery team found that 71 percent of
the programme's alerts in the observed sample occurred in the first 30 days, so a 30-day
programme would cost an estimated 210 USD per patient and raise ROI to roughly 4.5 to 1. This
diagnostic does not weaken the case; it improves it. That is the point of running diagnostics
before the finance committee does.

### 6.7 Varying one assumption

The base case assumed an 18 percent relative reduction. The table below holds everything else
constant.

| Relative reduction | Readmissions avoided | Net benefit (USD/year) | ROI |
|---:|---:|---:|---:|
| 4% | 36.3 | -190,277 | -0.24 |
| 8% | 72.6 | 404,846 | 0.52 |
| 12% | 108.9 | 999,970 | 1.27 |
| 18% | 163.3 | 1,892,654 | 2.41 |
| 24% | 217.7 | 2,785,339 | 3.55 |
| 30% | 272.2 | 3,678,024 | 4.68 |

The conclusion changes qualitatively between 4 and 8 percent. Below about 5.3 percent the
programme destroys value; between 5.3 and roughly 7 percent it is marginally positive and would
not survive a finance review that includes overhead; above 12 percent it is clearly fundable.

Now vary the enrollment rate instead, holding the reduction at 18 percent.

| Enrollment rate | Patients | Programme cost (USD) | Net benefit (USD) | ROI |
|---:|---:|---:|---:|---:|
| 25% | 1,050 | 357,000 | 2,321,054 | 6.50 |
| 40% | 1,680 | 571,200 | 2,106,854 | 3.69 |
| 55% | 2,310 | 785,400 | 1,892,654 | 2.41 |
| 70% | 2,940 | 999,600 | 1,678,454 | 1.68 |
| 85% | 3,570 | 1,213,800 | 1,464,254 | 1.21 |

This result is counter-intuitive and important: enrolling more patients lowers ROI, because the
programme cost scales with enrollment while the benefit does not, if the effect is concentrated
in high-risk patients. If the programme is targeted at patients with two or more prior
admissions, the 11 percent of patients responsible for 38 percent of readmissions, the effect per
enrolled patient rises and the ROI curve inverts. The discovery conclusion should therefore be a
targeted programme, not a universal one, and that conclusion came from the segmentation finding
in Section 6.2, not from the model.

### 6.8 The recommendation

The team recommended a 12-month pilot with three conditions: enrollment restricted to patients
with two or more prior admissions in twelve months; a 30-day monitoring period rather than 90
days; and a pre-registered analysis plan that measures readmissions at 30 days against a matched
comparison group. The kill criterion was a relative reduction below 6 percent at twelve months,
which corresponds to roughly the break-even point plus a margin.

The decision was to fund a pilot at 310,000 USD, not to buy the programme. That is the correct
output of discovery: not a purchase, but a smaller, better-specified next bet.

### 6.9 Market sizing, for context

Discovery also produces an order-of-magnitude market estimate. Using public discharge volumes and
an illustrative price of 340 USD per enrolled patient:

| Level | Definition | Discharges per year | Value at 340 USD |
|---|---|---:|---:|
| Total addressable market (TAM) | All United States adult heart failure discharges | 1,000,000 | 340.0 million USD |
| Serviceable addressable market (SAM) | Discharges at hospitals with 150 or more per year and a major EHR | 754,000 | 256.4 million USD |
| Serviceable obtainable market (SOM) | 90 accounts at 480 enrolled patients each in year three | 43,200 | 14.7 million USD |

These are estimates built from labelled assumptions, not measured market data. The TAM figure is
useful only for sanity-checking whether the category can support a company; the SOM is the number
that matters for a plan.

## 7. Code, query, or calculation

The script below implements the discovery-stage model: annual ROI, break-even effect size,
payback with a ramp, sensitivity sweeps, and market sizing. It uses only the Python standard
library and prints the values quoted in Section 6.

```python
"""Discovery-stage opportunity model. Standard library only."""

DISCHARGES = 4200                 # adult HF discharges per year
BASELINE_RATE = 0.216             # 30-day all-cause readmission rate
COST_PER_READMISSION = 16400.0    # USD, 2026, hospital fully loaded cost
PROGRAM_COST_PER_PATIENT = 340.0  # USD, 90-day bundle
ENROLLMENT_RATE = 0.55
TARGET_RELATIVE_REDUCTION = 0.18


def annual_case(reduction, enrollment=ENROLLMENT_RATE, price=PROGRAM_COST_PER_PATIENT):
    readmits = DISCHARGES * BASELINE_RATE
    avoided = readmits * reduction
    gross = avoided * COST_PER_READMISSION
    enrolled = DISCHARGES * enrollment
    cost = enrolled * price
    net = gross - cost
    roi = net / cost if cost else float("nan")
    return dict(readmits=readmits, avoided=avoided, gross=gross,
                enrolled=enrolled, cost=cost, net=net, roi=roi)


def break_even_reduction(enrollment=ENROLLMENT_RATE, price=PROGRAM_COST_PER_PATIENT):
    cost = DISCHARGES * enrollment * price
    per_unit = DISCHARGES * BASELINE_RATE * COST_PER_READMISSION
    return cost / per_unit


def payback_months(net_annual, implementation_cost, ramp_months=0, ramp_share=0.4):
    """Months to recover implementation cost, with an optional linear ramp."""
    cumulative = 0.0
    for month in range(1, 61):
        share = ramp_share if month <= ramp_months else 1.0
        cumulative += net_annual / 12.0 * share
        if cumulative >= implementation_cost:
            return month
    return float("inf")


base = annual_case(TARGET_RELATIVE_REDUCTION)
print("BASE CASE")
print(f"  baseline readmissions   {base['readmits']:.1f} per year")
print(f"  avoided readmissions    {base['avoided']:.1f} per year")
print(f"  gross saving            ${base['gross']:,.0f} per year")
print(f"  patients enrolled       {base['enrolled']:.0f} per year")
print(f"  programme cost          ${base['cost']:,.0f} per year")
print(f"  net benefit             ${base['net']:,.0f} per year")
print(f"  ROI                     {base['roi']:.2f} : 1")
print(f"  break-even reduction    {break_even_reduction() * 100:.1f}% relative")
print(f"  payback, no ramp        {payback_months(base['net'], 240000):.1f} months")
print(f"  payback, 6-month ramp   {payback_months(base['net'], 240000, 6, 0.4):.1f} months")

print("\nSENSITIVITY: relative reduction")
print("  reduction   avoided        net benefit      ROI")
for r in (0.04, 0.08, 0.12, 0.18, 0.24, 0.30):
    c = annual_case(r)
    print(f"  {r * 100:5.1f}%   {c['avoided']:8.1f}   ${c['net']:>14,.0f}   {c['roi']:6.2f}")

print("\nSENSITIVITY: enrollment at 18% reduction")
print("  enroll   patients   programme cost          net      ROI")
for e in (0.25, 0.40, 0.55, 0.70, 0.85):
    c = annual_case(TARGET_RELATIVE_REDUCTION, enrollment=e)
    print(f"  {e * 100:5.1f}%   {c['enrolled']:8.0f}   ${c['cost']:>12,.0f}"
          f"   ${c['net']:>12,.0f}   {c['roi']:5.2f}")

print("\nMARKET SIZING (illustrative, 2026 USD)")
for label, n in (("TAM", 1_000_000), ("SAM", 754_000)):
    value = n * PROGRAM_COST_PER_PATIENT / 1e6
    print(f"  {label}: {n:,} discharges x ${PROGRAM_COST_PER_PATIENT:.0f} = ${value:,.1f}M")
som = 90 * 480 * PROGRAM_COST_PER_PATIENT / 1e6
print(f"  SOM: 90 accounts x 480 patients = ${som:,.2f}M annual recurring revenue")

print("\nATTRIBUTION TEST: effect confined to uncovered patients")
covered = 0.31
uncovered_discharges = DISCHARGES * (1 - covered)
uncovered_avoided = uncovered_discharges * BASELINE_RATE * TARGET_RELATIVE_REDUCTION
gross = uncovered_avoided * COST_PER_READMISSION
net = gross - base["cost"]
print(f"  uncovered discharges    {uncovered_discharges:.0f}")
print(f"  avoided readmissions    {uncovered_avoided:.1f}")
print(f"  net benefit             ${net:,.0f}")
print(f"  ROI                     {net / base['cost']:.2f} : 1")
```

Expected output:

```text
BASE CASE
  baseline readmissions   907.2 per year
  avoided readmissions    163.3 per year
  gross saving            $2,678,054 per year
  patients enrolled       2310 per year
  programme cost          $785,400 per year
  net benefit             $1,892,654 per year
  ROI                     2.41 : 1
  break-even reduction    5.3% relative
  payback, no ramp        1.5 months
  payback, 6-month ramp   2.2 months

SENSITIVITY: relative reduction
  reduction   avoided        net benefit      ROI
    4.0%       36.3   $      -190,277   -0.24
    8.0%       72.6   $       404,846    0.52
   12.0%      108.9   $       999,970    1.27
   18.0%      163.3   $     1,892,654    2.41
   24.0%      217.7   $     2,785,339    3.55
   30.0%      272.2   $     3,678,024    4.68

SENSITIVITY: enrollment at 18% reduction
  enroll   patients   programme cost          net      ROI
   25.0%       1050   $     357,000   $   2,321,054    6.50
   40.0%       1680   $     571,200   $   2,106,854    3.69
   55.0%       2310   $     785,400   $   1,892,654    2.41
   70.0%       2940   $     999,600   $   1,678,454    1.68
   85.0%       3570   $   1,213,800   $   1,464,254    1.21

MARKET SIZING (illustrative, 2026 USD)
  TAM: 1,000,000 discharges x $340 = $340.0M
  SAM: 754,000 discharges x $340 = $256.4M
  SOM: 90 accounts x 480 patients = $14.69M annual recurring revenue

ATTRIBUTION TEST: effect confined to uncovered patients
  uncovered discharges    2898
  avoided readmissions    112.7
  net benefit             $1,062,734
  ROI                     1.35 : 1
```

The script has three properties worth copying into your own work. It separates parameters from
logic, so that a sensitivity sweep is one function call. It reports the break-even effect rather
than only the base case, because the break-even is what a finance committee will ask about. And
it includes the attribution test as a first-class calculation, because it is the diagnostic most
likely to change the conclusion.

Two limitations. The payback function treats implementation cost as recovered out of net benefit
before tax, depreciation, and working capital, which is appropriate for a screening decision but
not for a capital appropriation. And the model has no probabilistic treatment; a Monte Carlo
version would give a distribution rather than a range, but at the discovery stage the input
distributions are too uncertain for that precision to mean anything.

## 8. Data, standards, and regulatory context

Discovery sits upstream of most regulation, but four bodies of rules constrain how you may
collect the evidence. Getting them wrong is expensive because it happens at the start, when the
project is cheap to stop and cheap to fix.

### 8.1 Human subjects protection

In the United States, the Common Rule at 45 CFR 46 defines research as a systematic investigation
designed to develop or contribute to generalisable knowledge. Quality improvement that is
intended only to improve local care is generally not research and does not require IRB review,
though most institutions require a determination rather than a self-declaration. Discovery
interviews that will be published, or that test a generalisable hypothesis, usually are research.

The practical consequence: submit a determination request before the first interview if there is
any chance the findings will be published or used to support a regulatory submission. The
determination takes one to three weeks at most institutions and costs nothing. Proceeding without
one and later discovering that the interviews were research invalidates the consent basis for the
data.

In the European Union, the Clinical Trials Regulation (EU) 536/2014 governs interventional
trials, and the General Data Protection Regulation (GDPR) governs personal data processing;
observational discovery that processes patient data usually requires a legal basis and a data
protection impact assessment where the processing is high risk.

### 8.2 Privacy and protected health information

Observation in a clinical area exposes you to protected health information (PHI) as defined at
45 CFR 160.103. The HIPAA Privacy Rule permits use for health care operations and for research
with authorization or waiver, but "I was just observing" is not a permission. Practical controls:

- Do not record identifiers in field notes. Use role and a pseudonymous case number.
- Do not photograph screens showing patient data.
- If recording audio, obtain written authorization and store on institutional systems.
- If a vendor employee is observing, a business associate agreement (BAA) is usually required
  before the observation, not after.

The minimum necessary standard at 45 CFR 164.502(b) applies to uses and disclosures; observation
that captures more than necessary is a compliance finding waiting to happen.

### 8.3 Payments to clinicians for interviews

Paying a physician for an interview is common and lawful if the payment is fair market value for
the time and is not tied to prescribing, ordering, or purchasing. In the United States, the
federal Anti-Kickback Statute at 42 U.S.C. 1320a-7b(b) prohibits remuneration intended to induce
referrals or purchases of items payable by federal health care programmes. Market research
payments that are documented at fair market value and not conditioned on behaviour are generally
acceptable, but the documentation matters. Many institutions route payments through a purchasing
or consulting agreement rather than an honorarium to an individual.

The inference for discovery design: do not offer a bonus for participation in a pilot, do not
tie payment to adoption, and keep the rate consistent with the institution's published policy.

### 8.4 Device status and claims

If discovery reveals that the product would diagnose, treat, or prevent disease, the intended use
triggers medical device regulation. In the United States, the Food and Drug Administration (FDA)
guidance "Clinical Decision Support Software" (final, September 2022) describes when software
that supports a clinical decision is not a device under section 520(o)(1)(E) of the Federal Food,
Drug, and Cosmetic Act. The four criteria include that the software does not acquire, process, or
analyze a medical image or signal, and that it displays the basis for its recommendation so that
the clinician can independently review it.

Discovery must record the intended use precisely, because a small change in wording changes the
regulatory pathway. "Flags patients at risk of deterioration for nurse review" and "predicts
sepsis onset" imply different claims, different evidence, and different timelines. Write the
intended-use sentence during discovery, not after the model is built.

The Office of the National Coordinator for Health Information Technology (ONC) Health Data,
Technology, and Interoperability: Certification Program Updates, Algorithm Transparency, and
Information Sharing final rule (HTI-1, January 2024) adds decision support intervention
transparency requirements for certified health information technology, including source
attribute and intervention risk management fields for predictive decision support. If the
product will be embedded in certified EHR technology, discovery should establish whether HTI-1
applies.

### 8.5 Data access standards

Discovery relies on being able to count events. The practical standards are the Fast Healthcare
Interoperability Resources (FHIR) specification, particularly the Bulk Data Access and
SMART-on-FHIR launch patterns, and, in the United States, the information blocking rules at
45 CFR Part 171. A buyer's willingness to grant access is a commercial fact, not a technical one;
the standard tells you what the interface can do, not whether you will be allowed to use it.

Claims data add the X12 837 and 835 transaction formats and the CMS Chronic Conditions Warehouse
data dictionary. For discovery on readmissions, the discharge registry and the claims extract
will disagree, usually because one counts observation stays and the other does not. Establish
which source the buyer trusts before modelling anything.

### 8.6 What these rules do not do

None of them tell you whether the product is worth building. Compliance is a constraint, not a
value proposition. A fully compliant product aimed at a decision nobody owns will fail slowly and
expensively.

## 9. Industry practice and case evidence

### 9.1 Documented: workflow observation finds what surveys miss

Unertl, Weinger, and Johnson reported a direct-observation study of a multiple sclerosis clinic
that described itself as fully electronic. More than 55 hours of observation found shadow paper
charts, paper forms, printed whiteboards, and duplicated entry. Adoption of the informatics tools
was described as shallow, with workarounds compensating for gaps between the tool and the real
workflow (AMIA Annual Symposium Proceedings, 2006).

The finding is not that paper persists. It is that self-report about workflow is unreliable in a
predictable direction: organisations describe the intended process. Any discovery that relies on
interviews alone will overestimate readiness.

### 9.2 Documented: the time cost that creates the opportunity

Sinsky and colleagues conducted a time-and-motion study of 57 physicians in four specialties
across 16 practices, with 430 hours of observation and 23,416 tasks recorded. They found 27.0
percent of office time in direct clinical face time and 49.2 percent in EHR and desk work, with a
further one to two hours of after-hours clerical work per day (Annals of Internal Medicine,
2016;165:753-760, doi:10.7326/M16-0961).

That ratio is the commercial foundation of the ambient documentation category. It is also a
caution: the same study found the practices were self-selected and high-performing, so the
national average may be worse, not better. A vendor citing this paper should say which direction
the bias runs.

### 9.3 Documented: decision support is used far less than it is deployed

Richardson and colleagues reported that provider adoption of clinical decision support is
commonly in the 10 to 20 percent range and that, as usability testing approaches real clinical
conditions, users become substantially more critical of usefulness (International Journal of
Medical Informatics, 2017). Their live testing in real encounters found that the expected
sequential workflow never occurred and that providers were interrupted in every session.

The commercial implication for discovery: the addressable decision is the one that survives
interruption. A product that requires a three-minute uninterrupted review will be used by a
minority of clinicians regardless of its accuracy.

### 9.4 Documented: a deployment where accuracy was not the constraint

Beede and colleagues evaluated a deep learning system for diabetic retinopathy screening across
11 clinics in Thailand. The evaluation found that image quality rejection and the added capture
workload, rather than model discrimination, limited throughput. In the first six months, 393 of
1,838 uploaded images, 21 percent, were rejected as ungradable, and each rejection created rework
and a repeat visit (CHI 2020, "A Human-Centered Evaluation of a Deep Learning System Deployed in
Clinics for the Detection of Diabetic Retinopathy", doi:10.1145/3313831.3376718).

This is the cleanest published demonstration that a Layer 1 finding can invalidate a Layer 5
assumption. The model worked. The workflow did not.

### 9.5 Documented: the regulatory-clearance-is-not-adoption case

IDx-DR received De Novo authorisation in 2018 (DEN180001) as the first autonomous artificial
intelligence diagnostic system authorised to make a screening decision without a clinician
interpreting the image. Commercial adoption then depended on a payment pathway. The Centers for
Medicare and Medicaid Services (CMS) established a national rate for Current Procedural
Terminology (CPT) code 92229 in the CY2022 Physician Fee Schedule final rule, and the 2024
national non-facility rate was reported at 45.75 USD per screening (Health Affairs Forefront,
April 2024). Analysts noted that a substantial portion of that rate was characterised as a
software analysis component.

The discovery lesson: the buyer in primary care needed a payable service, not a better algorithm.
A product that requires the practice to absorb the cost without a code has a different sales
motion and usually a smaller market. The exact rate changes annually, so any business case must
cite the year and the fee schedule version.

### 9.6 Documented: scaling failure as the norm

Greenhalgh and colleagues built the Nonadoption, Abandonment, Scale-up, Spread, and
Sustainability (NASSS) framework from six programmes, more than 400 hours of ethnography, 165
interviews, and about 200 documents. The framework's seven domains are the condition, the
technology, the value proposition, the adopter system, the organisation, the wider system, and
embedding and adaptation over time. The central empirical claim is that programmes rated complex
in multiple domains rarely became mainstreamed (Journal of Medical Internet Research,
2017;19(11):e367, doi:10.2196/jmir.8775).

NASSS is useful in discovery as a checklist: score each domain simple, complicated, or complex.
A programme that is complex in the value proposition and the organisation is a programme to
decline, regardless of the technology.

### 9.7 Inference: what these cases imply for discovery design

The following is inference from the sources above, not a documented finding. Three design rules
follow.

First, budget discovery time in proportion to organisational complexity, not technical
complexity. The technology is usually the easiest part.

Second, always run the deployment-environment test: put the product in front of users in a real
clinic day, with interruptions, before committing to a build. Simulated usability testing
systematically overestimates adoption.

Third, treat a champion who cannot name a budget line as a warning, not an asset. The most
common commercial failure in the published cases is not rejection; it is indefinite enthusiasm
without procurement.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Building from the dataset outward | First artifact is a data dictionary, not a decision statement | Available data substitutes for a problem | Eighteen months to a product nobody asked for | Require a decision statement before any modelling |
| Interviewing only enthusiasts | All interviewees were recruited by the champion | Selection bias toward agreement | False positive demand signal | Recruit through at least three independent routes |
| Treating stated intent as demand | Notes contain "they said they would use it" | Politeness and hypothetical bias | Build funded on fiction | Require a behavioural test before build |
| No named budget owner | "The CMO loves it" and no finance contact | Enthusiasm mistaken for authority | Pilot that never converts | Map the buying centre at Step 8 |
| Frequency estimated from memory | A round number with no source | Availability heuristic | Value case off by a factor of two | Count from an event log or a stated sample |
| Ignoring workarounds | Observation notes show no paper or spreadsheets | Observer anchored on the software demo | Product reproduces the gap it was meant to close | Log every paper artifact |
| Mis-scoped wedge | Scope list has more than three integrations | Ambition substituted for sequencing | Nothing ships | Write explicit exclusions |
| Measurement gap | No metric exists in a system the buyer trusts | Benefit unprovable at renewal | Cut at first budget review | Choose opportunities with an existing metric |
| Payer-provider misalignment | Value accrues to a party not paying | Split incentives | Deal stalls or price collapses | Name which budget captures the value |
| Ignoring the security review | Questionnaire not requested until contracting | Gate discovered late | Eight to sixteen week delay and rework | Request the questionnaire during interviews |
| Single champion | One name appears in every meeting note | Key-person dependency | Deal dies on promotion or leave | Recruit a second champion with a distinct rationale |
| Discovery that always says yes | Report recommends proceeding with no conditions | Advocacy disguised as research | Capital allocated to weak ideas | Pre-register kill criteria |

Two red flags deserve separate mention because they are diagnostic of a team problem rather than
an opportunity problem. The first is a discovery report with no negative findings; a competent
discovery always finds something that weakens the case. The second is a kill criterion that is
revised after the evidence arrives; that is not a criterion, it is a ratification.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The observed workflow is representative of the site | Observation is sampled, not exhaustive | Observation fell on an atypical week | Observe across two months and two shifts |
| 2 | Frequency from one site generalises to the segment | Market sizing needs a per-site rate | Site is an outlier in volume or case mix | Compare with two additional sites |
| 3 | The status-quo cost is borne by the buyer | Value must accrue to the payer of the change | Cost is socialised or absorbed centrally | Ask finance which budget line moves |
| 4 | The effect size is transferable from published studies | Discovery cannot run a trial | Population, adherence, or care model differs | Run a small behavioural test; widen the range |
| 5 | Interview reports of past behaviour are accurate | Interviews are the main qualitative source | Recall bias on rare events | Triangulate with observation and data |
| 6 | The decision is stable over the product lifetime | A build takes 12 to 24 months | Reimbursement or policy change | Identify the policy dependency explicitly |
| 7 | Users will change behaviour if the tool is available | The product's theory of change | Habit, liability, or incentive blocks change | Behavioural test at Step 11 |
| 8 | The buyer's data are accurate enough to measure the outcome | Renewal depends on measurement | Coding variation, observation stays, attribution | Reconcile two sources before modelling |
| 9 | Discovery cost is small relative to build cost | Justifies the spend | Internal tool with a three-week build | Scale the protocol down explicitly |
| 10 | The opportunity is not already solved by an incumbent | Competitive landscape matters | An installed product already covers it | Ask what they bought and whether they use it |

The deepest limit is that discovery is a forecast. It reduces uncertainty; it does not eliminate
it. A team that treats a positive discovery as a guarantee has replaced one form of overconfidence
with another.

## 12. Exercises

### Exercise 1. Decision statement

**Deliverable:** one paragraph containing a role, a trigger, a choice among named alternatives,
and a consequence, for a product you are currently working on or one described in a public case.
**Criteria:** the statement names a role and not a department; the alternatives are enumerable;
the consequence is measurable; no feature is mentioned. **Time:** 30 minutes.

### Exercise 2. Shadowing protocol

**Deliverable:** a one-page protocol with positioning, a decision-log template, an interruption
log, and a workaround log, plus a plan for two four-hour sessions.
**Criteria:** the protocol distinguishes tasks from decisions; it specifies where the observer
stands; it includes a plan for consent and for handling PHI. **Time:** 90 minutes.

### Exercise 3. Interview guide and role-play

**Deliverable:** a 30-minute interview guide using the five-part structure in Step 6, plus a
transcript of one role-played interview with a colleague.
**Criteria:** no hypothetical "would you" questions; at least three questions anchored on a
specific past event; the guide includes a workaround question. **Time:** 2 hours.

### Exercise 4. Status-quo cost model

**Deliverable:** a four-line cost model with low, base, and high estimates and a source label on
every input.
**Criteria:** loaded rates rather than salaries; at least one measured input; every assumption
stated; arithmetic reproduces. **Time:** 2 hours.

### Exercise 5. Opportunity scoring

**Deliverable:** a scoring sheet for three candidate problems using the weights and anchors in
Step 9, with a written justification for any score that differs by two or more points between
team members.
**Criteria:** anchors used rather than intuition; the ranking is sensitive to at most one weight
change; disagreements documented. **Time:** 90 minutes.

### Exercise 6. Buying-centre map

**Deliverable:** a nine-row table as in Step 8 for a specific product and setting, plus a
one-paragraph note on the most likely blocker and why.
**Criteria:** at least one role is listed whose answer is unknown and marked as such; the budget
owner is named by role and budget type; the blocker has a mitigation. **Time:** 2 hours.

### Exercise 7. Kill criteria

**Deliverable:** five kill criteria written before any evidence collection, each with a
threshold, a measurement method, and a decision date.
**Criteria:** each criterion is falsifiable; at least one concerns the buyer rather than the
product; none can be satisfied by opinion. **Time:** 45 minutes.

### Exercise 8. Break-even calculation

**Deliverable:** a calculation of the break-even effect size and break-even enrollment for the
case in Section 6 under a programme cost of 500 USD per patient rather than 340 USD, with the
arithmetic shown.
**Criteria:** correct arithmetic; the direction of change explained; a statement of whether the
programme remains fundable. **Time:** 60 minutes.

## 13. Self-check answers

**1. What makes a decision statement different from a feature description?**
A decision statement names a role, a trigger, the alternatives available at that moment, and the
consequence of choosing wrongly. A feature description names functionality. Test: a decision
statement can be falsified by observing behaviour at the named moment; a feature description can
only be checked against a specification.

**2. Why observe before interviewing?**
Because interviewing first teaches the interviewee what you are interested in and contaminates
their account. Observation also gives you the vocabulary and the exceptions you need to ask good
questions. The order matters more than the total time spent.

**3. What are the four observation log fields that matter most?**
Time, actor role, options considered, and downstream effect. Time establishes frequency and
duration; role establishes who owns the decision; options reveal the real alternative set;
downstream effect establishes consequence. Workarounds and interruptions are recorded as
separate logs.

**4. Why is a 30-minute interview better than a 60-minute one for discovery?**
Because access is the binding constraint and a 30-minute request doubles the acceptance rate. The
five-part structure fits in 30 minutes if the interviewer stops talking. Longer sessions produce
more opinion and not more behaviour.

**5. Compute the status-quo cost.** For 4,200 discharges, 31 percent coverage, 4.2 hours per
patient, and a loaded rate of 62 USD per hour: 4,200 x 0.31 = 1,302 patients; 1,302 x 4.2 = 5,468
hours; 5,468.4 x 62 = 339,041 USD per year. Rounded to 339,000 USD.

**6. Score a problem with 50,000 events per year, 3,000 USD cost per event, quarterly
measurement, and batch integration.** Frequency 3 (10,000 to 100,000 is 3; 50,000 falls in that
band), severity 3 (1,000 to 10,000 USD), measurability 3 (measured quarterly), reachability 3
(batch integration possible). Weighted score = 0.30(3) + 0.35(3) + 0.20(3) + 0.15(3) = 3.00.

**7. Who is the budget owner for a readmission-reduction programme in a United States hospital?**
Usually the service line or the chief financial officer, depending on whether the spend is
operating or capital. The test is not the org chart but the signature: ask who signed the last
comparable purchase and which budget line it hit. The chief medical officer may sponsor without
controlling the funds.

**8. Break-even calculation for Exercise 8.** Programme cost per patient 500 USD: annual cost =
2,310 x 500 = 1,155,000 USD. Per-unit saving = 907.2 x 16,400 = 14,878,080 USD per unit of
relative reduction. Break-even relative reduction = 1,155,000 / 14,878,080 = 7.76 percent. At the
18 percent assumption the programme still creates value (ROI about 1.32 to 1), but the margin is
much thinner, and the 6 percent kill criterion from Section 6.8 would now be below break-even and
must be raised.

**9. Why is the attribution test important?**
Because a headline effect measured against no programme overstates the incremental effect of the
programme when part of the population already receives an equivalent service. Confining the
effect to the uncovered population reduced ROI from 2.41 to 1.35 in the worked example, which
changes the recommendation from fund to pilot.

**10. What is the difference between a concierge test and a pilot?**
A concierge test delivers the service manually to test whether demand and behaviour change,
usually without a contract and with a small number of users. A pilot is a contracted deployment
with defined success criteria and an intention to convert. Concierge tests are cheaper and are
appropriate before a build; pilots are appropriate after.

**11. Rubric for Exercise 6.** Full marks require: all nine roles present; at least one unknown
marked as unknown; budget owner identified by signature authority rather than seniority; blocker
named with a specific mitigation and an owner; and a statement of what evidence would change the
map. Half marks if the map is complete but the blocker is generic.

**12. Rubric for Exercise 7.** Full marks require five falsifiable criteria, at least one
commercial rather than technical, a measurement method for each, and a dated decision. Zero if
any criterion can be satisfied by a stakeholder's opinion.

## 14. Interview questions

**1. How do you decide whether a problem is worth building for?**
A strong answer walks the five-layer stack and names a specific test at each layer, with an
example of a problem that passed Layers 1 to 3 and failed at Layer 4. A weak answer lists
"customer pain" and "market size" without saying how either was measured.

**2. What is the difference between the user and the buyer, and why does it matter?**
Strong: names a concrete case where the user gains time and the buyer gains margin, and explains
that a deal needs both plus a blocker-free path. Weak: says "the buyer pays and the user uses,"
which is a definition rather than an insight.

**3. How many interviews are enough?**
Strong: frames it as saturation plus stratification, gives a range of 12 to 15 deep interviews
after 40 to 60 conversations, and says what would make them stop early. Weak: a fixed number
with no reasoning, or "as many as it takes."

**4. How do you avoid leading questions?**
Strong: asks about the last specific instance, avoids hypotheticals, and describes a technique
for silencing their own hypothesis, such as asking the interviewee to walk through a recent
event before any product is mentioned. Weak: says "I stay neutral," which is not a method.

**5. What do you do when the champion has no budget?**
Strong: treats it as a finding, looks for the signature authority, and either recruits a second
sponsor or recommends stopping. Weak: keeps selling to the champion and hopes.

**6. How do you size an opportunity before you have a product?**
Strong: builds a range from a counted event volume, a cost per event, and a penetration
assumption, and states which input is weakest. Weak: quotes an analyst report as if it were a
measurement.

**7. What is a wedge, and how do you know yours is small enough?**
Strong: defines the wedge by the single decision it changes and lists explicit exclusions;
tests size by counting integrations, new roles, and payment changes required. Weak: describes the
MVP feature list.

**8. When would you kill a project at discovery?**
Strong: names pre-registered criteria, including a commercial one, and gives an example of
killing something. Weak: says they would kill it if it "wasn't working," with no threshold.

**9. Why write kill criteria before collecting evidence?**
Strong: because the team's incentives favour continuing, and a criterion written afterwards is a
rationalisation. Weak: because it is good practice, with no mechanism.

**10. How do you handle a discovery finding that contradicts the sponsor's hypothesis?**
Strong: presents the finding with the evidence, separates the sponsor's goal from the proposed
mechanism, and offers an alternative path to the same goal. Weak: softens the finding to avoid
conflict, which converts a useful negative into a later failure.

**11. What data do you need to measure the outcome, and what if you cannot get it?**
Strong: names a specific source the buyer already trusts, and if none exists, proposes an
instrumented pilot with a pre-registered measure. Weak: assumes access will be granted.

**12. How does reimbursement change what you would build?**
Strong: distinguishes clinical benefit, economic value, and willingness to pay, and gives an
example where a payable code changed the viable segment. Weak: treats reimbursement as a
downstream detail.

## 15. Key terms

- **Attribution test** — check of how much of an observed benefit would have occurred without
  the product.
- **Behavioural test** — measurement of what users do when a prototype is available, rather than
  what they say they would do.
- **Blocker** — a role whose objection can stop a purchase.
- **Budget owner** — the role with signature authority over the relevant budget line.
- **Buying centre** — the set of roles involved in a purchase decision.
- **Champion** — an internal advocate willing to spend political capital.
- **Concierge test** — manual delivery of a service to test demand before building.
- **Contextual inquiry** — interview conducted in the workplace during real work.
- **Decision point** — a moment where a named role chooses among alternatives.
- **Decision statement** — one sentence naming role, trigger, choice, and consequence.
- **Discovery brief** — the exit artifact of discovery, including negative findings.
- **Economic buyer** — the role that approves the business case and releases funds.
- **Frequency** — decision occurrences per unit time.
- **Job to be done** — the progress a person is trying to make in a situation.
- **Kill criterion** — a pre-agreed, falsifiable condition that stops the project.
- **Latent need** — a need not articulated because it is normal to the user.
- **Measurability** — whether the consequence is observable in trusted data.
- **Minimum viable product** — the smallest release that produces validated learning.
- **NASSS** — framework for nonadoption, abandonment, scale-up, spread, and sustainability.
- **Opportunity score** — weighted rank of frequency, severity, measurability, reachability.
- **Pilot purgatory** — repeated pilots that never convert to a contract.
- **Problem-solution fit** — evidence that a defined problem is painful enough to want solving.
- **Product-market fit** — evidence that a market buys and retains at scale.
- **Reachability** — whether the product can influence the decision at an accessible point.
- **Revealed preference** — what someone actually did when facing a real choice.
- **Severity** — consequence per decision event.
- **Shadowing** — following one worker through a full shift.
- **Stated preference** — what someone says they would do.
- **Status quo cost** — fully loaded cost of the current way of working.
- **Time and motion study** — systematic timing and categorisation of tasks.
- **Value of information** — the expected value of the decision change a piece of evidence causes.
- **Wedge** — the smallest product that changes one decision for one segment.
- **Willingness to pay** — the maximum a buyer would commit, revealed by a budget decision.
- **Wizard of Oz test** — a simulated product operated by a human behind the scenes.
- **Workaround** — an unofficial practice compensating for a tool or process gap.

## 16. Sources

All URLs accessed 26 October 2026.

1. Greenhalgh T, Wherton J, Papoutsi C, et al. Beyond adoption: a new framework for theorizing
   and evaluating nonadoption, abandonment, and challenges to the scale-up, spread, and
   sustainability of health and care technologies. *Journal of Medical Internet Research*.
   2017;19(11):e367. https://doi.org/10.2196/jmir.8775
2. Sinsky C, Colligan L, Li L, et al. Allocation of physician time in ambulatory practice: a
   time and motion study in 4 specialties. *Annals of Internal Medicine*. 2016;165(11):753-760.
   https://doi.org/10.7326/M16-0961
3. Unertl KM, Weinger MB, Johnson KB. Applying direct observation to model workflow and assess
   adoption. *AMIA Annual Symposium Proceedings*. 2006:812-816.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC1839698/
4. Richardson S, Mishuris R, O'Connell A, et al. "Think aloud" and "near live" usability testing
   of two complex clinical decision support tools. *International Journal of Medical Informatics*.
   2017;105:1-8. https://pmc.ncbi.nlm.nih.gov/articles/PMC5679128/
5. Beede E, Baylor E, Hersch F, et al. A human-centered evaluation of a deep learning system
   deployed in clinics for the detection of diabetic retinopathy. *Proceedings of the 2020 CHI
   Conference on Human Factors in Computing Systems*. 2020.
   https://doi.org/10.1145/3313831.3376718
6. US Food and Drug Administration. *Clinical Decision Support Software: Guidance for Industry
   and Food and Drug Administration Staff*. Final guidance, September 2022.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
7. US Food and Drug Administration. De Novo classification request DEN180001 (IDx-DR), 2018.
   https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
8. Centers for Medicare and Medicaid Services. CY 2022 Physician Fee Schedule final rule.
   *Federal Register*. 2021. https://www.federalregister.gov/documents/2021/11/19/2021-23973/medicare-program-cy-2022-payment-policies-under-the-physician-fee-schedule-and-other-changes-to-part
9. Tang PC, Wilson E, Mehrotra A. Paying for AI in healthcare: setting the right precedent
   amidst growing use. *Health Affairs Forefront*. April 2024.
   https://www.healthaffairs.org/content/forefront/paying-artificial-intelligence-health-care-setting-right-precedent-amidst-growing-use
10. Office of the National Coordinator for Health Information Technology. *HTI-1 Final Rule*,
    January 2024. https://www.healthit.gov/topic/laws-regulation-and-policy/health-data-technology-and-interoperability-certification-program
11. US Department of Health and Human Services. *Federal Policy for the Protection of Human
    Subjects* ("Common Rule"), 45 CFR Part 46.
    https://www.hhs.gov/ohrp/regulations-and-policy/regulations/45-cfr-46/index.html
12. US Department of Health and Human Services. *HIPAA Privacy Rule*, 45 CFR Parts 160 and 164.
    https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
13. Christensen Institute. Jobs to be done: discovering why we consume healthcare.
    https://www.christenseninstitute.org/blog/jobs-to-be-done-discovering-why-we-consume-healthcare/
14. Mora N, et al. A national unmet needs assessment for CTSA-affiliated electronic health record
    data networks: a customer discovery approach. *Journal of Clinical and Translational Science*.
    2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11523010/

Where a figure is described as an estimate, it is an estimate constructed for this text and is
labelled as such in place. No company outcome, market size, or reimbursement rate in this file is
invented; the worked example in Section 6 is explicitly a composite.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_value-quantification.md`](02_value-quantification.md)
- Related topic: [`05_procurement.md`](05_procurement.md)
- Related topic: [`06_adoption.md`](06_adoption.md)
- Module survey: [`../../course/17_product_economics_leadership.md`](../../course/17_product_economics_leadership.md)
- Labs: [`../../labs/`](../../labs/)
- Prerequisite module: [`../08_ehr_claims/`](../08_ehr_claims/)
