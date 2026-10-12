# Behavioral and Project Questions for AI Roles

**TL;DR** — Twenty-eight behavioral questions answered in full STAR-L format — Situation, Task, Action, Result, Learning — with the AI-specific detail that makes them credible: cache hit rates, recall numbers, blast radius, kappa, cost models. Behavioral rounds for AI roles probe six things: how you structure ambiguity, how you handle a system that fails silently, how you manage stakeholders who believe the hype, how you disagree and get overruled, how you weigh safety against speed, and whether you can explain a limitation without jargon. Every answer here is a shape you can adapt to your own experience — do not memorise them, transplant the structure.

**The STAR-L framework and why the L matters**

| Letter | What it covers | Time budget (2-min answer) |
|---|---|---|
| **S** | Situation — the context, in two sentences, no more | 15 s |
| **T** | Task — what YOU specifically owned | 15 s |
| **A** | Action — 3–4 concrete steps you took, in order | 60 s |
| **R** | Result — a number. Always a number | 20 s |
| **L** | Learning — what changed in how you work | 10 s |

Most candidates spend 90 seconds on Situation and run out of time before the Result. The fix is mechanical: two sentences of context, then get to what you did.

**The three things that make a behavioral answer land in an AI interview**

1. **A number in the Result.** "It went well" is not a result. "Recall went from 0.61 to 0.79 and support tickets dropped 23%" is.
2. **A failure mode you named.** Behavioral answers about AI systems are credible when you demonstrate you knew how the system could break — because that's the job.
3. **A learning that changed a default.** "I learned communication is important" is nothing. "I now attach explicit kill criteria to my own proposals so walking back is a checkpoint, not a confession" is a habit.

---

## Section A — Ambiguity and problem framing

### Q1. Tell me about a time you started an AI project with unclear requirements.

**What they're testing:** Whether you structure ambiguity or drown in it. The trap is describing the chaos rather than the method you imposed on it.

**S:** We wanted to add AI-powered search to our product, and stakeholders had three incompatible visions. Sales wanted "magic" search that understood everything. Engineering wanted something shippable in a quarter. Leadership wanted visible differentiation from competitors.

**T:** As tech lead I owned getting us from "AI search" as a slogan to a concrete first version we could build and learn from.

**A:** I stopped the feature debate and interviewed five stakeholders about the underlying need rather than the requested feature. The common thread turned out to be *time to find information*, not conversational interface. I then built three throwaway prototypes in a week — better keyword ranking, semantic search with embeddings over a 40,000-document corpus, and a conversational RAG version — and demoed each against real queries pulled from our support logs, with the engineering effort and the cost per thousand queries attached to each one. That reframed the conversation from "which features" to "what value at what cost," which is a conversation with a decidable answer.

**R:** We aligned on semantic search with a phase-two plan, shipped in six weeks, and user time-to-answer dropped 40% measured against the same query set we'd used in the demos. The phase-two buy-in was easy because the phase-one number existed.

**L:** Ambiguity usually means stakeholders are optimising for different things while using the same words. Showing concrete options with real trade-offs and a real number moves that faster than any amount of alignment discussion. I now start every ambiguous project by building the cheapest thing that produces a number.

### Q2. How do you decide what to build when the path isn't obvious?

**What they're testing:** Whether you have a decision procedure or a personality. The trap is describing intuition as method.

**S:** We had three candidate uses for an LLM in our internal tooling — a documentation assistant, a ticket-triage classifier, and a code-review helper — and enough budget for one.

**T:** I owned the recommendation to engineering leadership.

**A:** I scored each on four axes with explicit criteria rather than vibes: value if it works (how many people, how often, how much time saved), probability it works (how much of the task is within current model capability), cost to build and run, and **blast radius if it's wrong**. That last axis was the differentiator. Code review has high value and moderate probability, but a confidently wrong review comment that a human accepts is a production defect with a long tail. Ticket triage has lower individual value but a bounded blast radius, a cheap fallback to the existing manual router, and it generates labelled data as a byproduct — which raises the probability the next project works. Documentation assistance scored well on everything but had the least measurable value.

**R:** We built the triage classifier first. It hit 84% agreement with the human router, which was enough to auto-route the confident 60% and queue the rest, and it produced 12,000 labelled examples in three months that made the code-review project substantially easier when we did it.

**L:** I learned to add blast radius as a first-class axis, not a footnote. It changed the ranking, because the option with the highest ceiling was also the one where a failure was most expensive and least visible. I now score every proposal on value, probability, cost, and blast radius before anything gets built.

### Q3. Describe a situation where you changed direction mid-project.

**What they're testing:** Whether you can abandon sunk cost on evidence, or whether you defend the plan. The trap is a story where you changed direction for a political reason rather than a data one.

**S:** Six weeks into a three-month project building a fine-tuned classifier for support ticket routing, we had a working fine-tune at 79% accuracy and a competing prompt-only approach at 76% that took two days to build.

**T:** I owned the delivery date and the quality bar of 85%.

**A:** The fine-tune had plateaued — three consecutive training runs produced 78–79% with different hyperparameters, which told me the ceiling was the data, not the training. The prompt-only approach was still climbing as we added few-shot examples and better retrieval of similar past tickets. I set an explicit decision date rather than letting it drift: two more weeks, and if the fine-tune didn't beat the prompt approach by five points, we'd switch. At the decision date the fine-tune was at 80% and the prompt approach was at 83%, still climbing. I wrote up the comparison, recommended the switch, and we committed to the prompt path plus a retrieval layer over historical resolutions.

**R:** We shipped two weeks after the original date at 87% accuracy — above the 85% bar — and the ongoing cost was a per-request LLM call rather than a GPU, which turned out to be cheaper at our volume. The fine-tune work wasn't wasted: the labelled dataset it required became the evaluation set the prompt approach was measured against.

**L:** Setting a decision date in advance is what made this possible. Without it, "two more weeks" becomes "one more month" indefinitely, because there's always a plausible reason to keep going. I now write kill criteria into any proposal before I start, including my own, so switching is a checkpoint rather than an admission.

### Q4. Tell me about the most complex AI project you worked on.

**What they're testing:** Whether "complex" means technically intricate or genuinely hard, and whether you can describe your own contribution without claiming the whole team's work.

**S:** A multi-tenant RAG system over 4 million documents serving 500 enterprise tenants, where the hard part was never the retrieval — it was that every tenant had different documents, different access rules, different vocabulary, and different definitions of a correct answer.

**T:** I owned the retrieval and evaluation layers and the multi-tenancy architecture.

**A:** Three things made it tractable. First, I separated the corpus into a siloed index for the 20 largest tenants and a pooled index with a mandatory tenant predicate for the remaining 480 — which was a cost decision, and it cut the infrastructure bill by roughly 70% against siloing everyone. Second, I built the ACL into the chunk at ingest time rather than filtering at query time, so the access rule couldn't be forgotten in a code path. Third, and this is the part that actually mattered, I built a per-tenant evaluation set: 50 queries each, hand-labelled by the tenant's own domain expert. That converted "the system is roughly right" into a number per tenant, which turned arguments about quality into a work queue.

**R:** Retrieval nDCG@10 went from 0.58 to 0.79 over four months, the cross-tenant retrieval count stayed at zero across the whole period (it's a paging alert), and we onboarded the last 200 tenants without adding infrastructure because the pooled model scaled.

**L:** Complexity in AI systems is rarely a modelling problem — it's a diversity problem. Five hundred tenants with different notions of correct is harder than one task with a hard model, and the tool that makes it tractable is per-slice measurement, not a better model. I now build the per-slice eval set before I optimise anything, because otherwise I'm optimising an average that describes nobody.

---

## Section B — Failure and accountability

### Q5. Tell me about an AI system that did not work as expected in production.

**What they're testing:** Honesty plus systems thinking. The trap is a story where the failure was someone else's fault.

**S:** We launched a content recommendation system that beat our heuristic baseline in offline testing and then underperformed it by 30% on engagement in production.

**T:** I built it, so I owned the diagnosis, the rollback decision, and the stakeholder conversation.

**A:** I flagged it to leadership within a day with the raw numbers rather than with a theory, and proposed keeping 10% of traffic on the new system so I could keep investigating while the other 90% went back to the baseline. The investigation found two causes. Our test set was drawn from high-engagement users, so it wasn't representative — the offline evaluation was measuring the people most likely to engage with anything. And cold-start performance for new users was far worse than modelled, because the model leaned on history that new users didn't have. I rebuilt the evaluation with stratified sampling matching the production mix, added a heuristic fallback for users with fewer than five interactions, and built a live dashboard so stakeholders could watch the metric themselves rather than waiting for my summary.

**R:** The revised system outperformed the baseline by 15% after two more weeks of iteration. The stratified evaluation practice caught a similar distribution mismatch on the next project before it shipped.

**L:** Production is the only real test, and offline metrics that don't match the production population are worse than no metrics because they create false confidence. I now instrument before launch and plan for rapid iteration as the default, not as a contingency. And being transparent during the failure built more trust with the stakeholders than a clean launch would have.

### Q6. How do you handle a model that performs poorly in production?

**What they're testing:** Whether you have a diagnostic procedure or you start tuning prompts. The trap is jumping to a fix before you've located the failure.

**S:** Our support RAG assistant's answer quality dropped from a steady 4.3 out of 5 to 3.6 over about ten days. There had been no release, no prompt change, and no index rebuild.

**T:** I owned the investigation and the fix.

**A:** My first move was to resist the obvious hypothesis. I ran the daily golden set and confirmed the drop was real and reproducible, not judge noise. Then I separated the possible causes in a fixed order — model, prompt, index, judge — because any of them produces the identical symptom. The prompt and index were unchanged, confirmed from the release manifest. So it was the model or the judge. I compared the golden set score against the judge score across the same period and found the golden set (deterministic assertions) had barely moved while the judge score had fallen sharply — which meant the judge had drifted, not the system. It turned out a provider update to the judge model had made it markedly stricter on one rubric dimension.

**R:** We recalibrated the judge against a fresh human-labelled sample, kappa went from 0.38 back to 0.64, and the measured quality score recovered to 4.2. Total user impact: none — we'd caught a measurement problem, not a product problem. But we'd have spent a week tuning prompts against a phantom.

**L:** Always check whether the measurement moved before assuming the thing being measured moved. That's now step one in the runbook, not step five. And it's why the deterministic golden set and the judge score both exist — they fail differently, and the disagreement between them is itself the diagnostic.

### Q7. Describe a time you shipped something that had issues.

**What they're testing:** Whether you own it. The trap is minimising the issue, which reads as minimising the customer's problem.

**S:** We shipped a semantic cache in front of our support assistant to cut cost, and it worked — hit rate hit 38% and the bill dropped proportionally. Three days later a customer reported being told the refund window for their Enterprise plan was 30 days when it was 14.

**T:** I owned the cache design and the decision to ship it.

**A:** I disabled the cache within the hour — the whole feature, not the suspected key, because I didn't yet know the scope of the problem. Then I scoped it: I replayed 2,000 cached hits against the live model and found a 1.8% false-hit rate, concentrated in queries that differed only by plan tier. My cache key had included the tenant but not the plan tier, so a Pro-plan answer could be served to an Enterprise question — semantically near-identical strings, materially different answers. I also found that the cache wasn't keyed on the release version, so a prompt change wouldn't have invalidated it. I added both dimensions to the key, tightened the similarity threshold from 0.92 to 0.96, halved the TTL, and — the part I'd actually defend — added a false-hit monitor that samples cache hits and re-runs them fresh, because the false-hit rate was otherwise completely invisible.

**R:** Re-shipped two weeks later at a 31% hit rate and a measured 0.2% false-hit rate. The savings were lower and the system was correct, which was the right trade. The sampling monitor is now standard for anything we cache.

**L:** A cache is a correctness component that looks like a cost component, and the failure is silent by construction because the answer comes back with a 200 and no trace of where it came from. I now write down the full key dimensions before building any cache and check them against "what could change the answer." Missing the plan tier was an obvious omission in hindsight and completely invisible at the time.

### Q8. Tell me about a time you were wrong and had to walk it back.

**What they're testing:** Whether conviction survives evidence. The trap is a story where you were technically right and everyone else was wrong.

**S:** I pushed hard to replace our retrieval pipeline with a pure long-context approach — load the whole knowledge base into the prompt and drop the vector database. I argued it in two design reviews on recall and architectural simplicity, and the team committed a quarter to the migration partly on my conviction.

**T:** Two months in, I owned proving the new system out before full cutover.

**A:** The eval results were uncomfortable. Quality matched the retrieval system within noise. Cost ran four times projection, because cache hit rates collapsed under our update frequency — every corpus change invalidated a large prefix. And p95 latency doubled. My first instinct was to tune my way out, and after two weeks of marginal gains I accepted that the economics weren't going to close. I wrote a one-page memo with the numbers, stated plainly that my recommendation had been wrong on the economics rather than wrong on quality, and proposed a hybrid: long-context for the small static corpora where it genuinely shone, retrieval everywhere else. I presented it to the same audience I had originally convinced.

**R:** We kept the hybrid and salvaged about 60% of the migration work, because the ingestion and normalisation changes were useful either way. The postmortem produced a rule the team still uses: any architecture pitch must include a cost model under realistic update patterns, not just a quality benchmark.

**L:** Conviction is useful for getting decisions made and dangerous for unmaking them, because you've spent credibility on the position. I now attach explicit kill criteria to my own proposals before I make them, so walking back is a checkpoint I planned for rather than a confession I have to make.

### Q9. Describe a project that failed. What happened and what did you learn?

**What they're testing:** Whether you can distinguish a failed project from a failed person, and whether the learning is structural. The trap is picking a project that only half-failed.

**S:** We spent a quarter building a custom fine-tuned model for document summarisation. It worked — 4.4 out of 5 on our internal evaluation — and nobody used it.

**T:** I was the engineer on it and the person who had advocated for it.

**A:** The failure wasn't the model. It was that we'd never validated demand, and when I looked at adoption I found three structural reasons. The summaries were good but slower than the user's existing workflow of skimming the document themselves for short documents. The output format didn't match the template they pasted into their reports, so there was a manual reformatting step. And the feature lived in a separate tool rather than in the document viewer they already had open. I ran six user interviews, none of which we'd done before building, and all three causes came up in the first three conversations. I wrote it up as a postmortem rather than quietly letting it die, because the pattern was generalisable.

**R:** The model itself was repurposed into a batch summarisation job for an internal corpus where the format didn't matter, which recovered maybe 20% of the quarter. More usefully, we changed the process: any AI feature proposal now needs five user conversations and a named integration point before it gets engineering time.

**L:** A model that works is not a feature that works. I had confused model quality with product fit, and the evaluation I'd built measured the thing I could measure rather than the thing that mattered. Now, before I build, I ask what the user's current workflow is and where in it this lands — and if I can't answer both, I haven't done enough work to start.

---

## Section C — Managing expectations and stakeholders

### Q10. How do you explain AI limitations to non-technical colleagues?

**What they're testing:** Whether you can translate without condescending. The trap is either jargon or over-simplification that hides the real constraint.

**S:** Sales had committed an AI assistant to a prospect that would "answer any question about their contracts" — a corpus of 60,000 contracts across 14 jurisdictions with wildly inconsistent formatting and scanned attachments.

**T:** I owned setting realistic expectations without killing the deal.

**A:** I did it with a demo rather than a deck. I took twenty real questions from their own contract set, ran them live, and showed the answers with the source clauses. Seventeen were excellent and three were wrong, and I showed those three too rather than cherry-picking. Then I explained the three failure modes in plain language: scanned documents that hadn't parsed cleanly, questions that needed joining facts from two contracts, and questions where the contract was genuinely ambiguous and a lawyer would also hesitate. I gave them a concrete framework instead of a caveat: the assistant drafts, a human confirms, and the confirmation step is fast because the source clause is right there.

**R:** The deal closed with a modified scope — three clean jurisdictions first, expanded after the parse quality was verified. The three-bad-answers demo did more for trust than a polished one would have, because the prospect had already been burned by a vendor who'd only shown the good cases.

**L:** Show the failures before the customer finds them. A limitation you demonstrate is a property of the system; a limitation they discover is broken trust. And translate into workflow rather than capability — "the assistant drafts and you confirm" is something a business person can evaluate, whereas "90% accuracy" is a number they'll interpret optimistically.

### Q11. Tell me about a time you had to say no to a stakeholder request.

**What they're testing:** Whether you say no with an alternative or just a refusal. The trap is framing the no as a constraint you're reporting rather than a judgement you're making.

**S:** A product director wanted agentic write access to our CRM — the assistant should be able to update customer records directly, not just read them, because the copy-paste step was annoying users.

**T:** I owned the decision on tool permissions, and I disagreed with the request as framed.

**A:** I didn't argue against write access. I asked what the copy-paste step actually cost, and we measured it: about 40 seconds per interaction. Then I offered a middle path that captured most of the value at a fraction of the risk — the agent could prepare the record update, show a diff, and the user clicks once to apply it. Same number of clicks as copy-paste in the worst case, one click in the common case, and the agent never had write credentials. I also agreed to revisit full write access once we had three months of the propose-and-confirm data, because that data would tell us whether the agent's proposed updates were actually correct — which is the question that determines whether autonomous write access is safe.

**R:** We shipped propose-and-confirm, the manual step dropped from 40 seconds to under 5, and after three months the proposed updates were 97% accepted without edit. We then scoped write access for the two update types with the highest acceptance rate, with an audit log and a velocity limit.

**L:** "No" is rarely the right answer on its own; "not this way, and here's how we'd earn it" is. And the measurement that made the second decision possible was a byproduct of the first design, which is usually true — a conservative first version generates the evidence that unlocks the ambitious second version.

### Q12. Describe a situation where expectations were unrealistic.

**What they're testing:** Whether you correct the expectation with evidence or absorb it and fail later. The trap is a story where you heroically delivered the unrealistic thing, which teaches the wrong lesson.

**S:** Leadership had committed to a customer that our AI agent would resolve 60% of support tickets without human involvement, within one quarter. Our current automated resolution rate was 11%.

**T:** I owned the engineering plan and had to either commit to 60% or explain why not.

**A:** I built the estimate from the bottom up rather than arguing about the top-line number. I sampled 500 recent tickets, categorised them by how much judgement they needed, and found that roughly 45% were straightforward and within current capability, 30% needed account context we didn't have access to, and 25% were genuinely ambiguous or emotionally charged. That gave me a defensible ceiling: about 45% was what the current architecture could reach, and getting past it needed integrations, not a better model. I proposed two phases — hit 35% in the quarter with the straightforward categorisation, then revisit after the account-context integration landed — and I framed the second phase's number as dependent on that integration rather than on model improvement.

**R:** We hit 38% by end of quarter against the 35% commitment, which was better than the 11% starting point by a wide margin, and the customer was satisfied because we'd reset the expectation early with the ticket analysis rather than in week twelve. The account-context integration shipped the following quarter and took us to 52%.

**L:** When a number is unrealistic, don't argue about the number — build the estimate from the data and show the ceiling. The 500-ticket sample took me two days and it was the entire answer. And phase your commitments so the later number depends on a named dependency rather than on optimism; that makes the conversation about whether to build the dependency, which is a decidable question.

### Q13. How do you handle a stakeholder who wants to skip evaluation to ship faster?

**What they're testing:** Whether you hold a line without becoming an obstacle, and whether you can price the thing they want to skip.

**S:** Two weeks before a launch, a product manager asked me to skip the golden-set evaluation run because it was "adding three days" and the release date was fixed by an event.

**T:** I owned the release quality gate and had to either hold it or find another path.

**A:** Rather than arguing about process, I priced the risk. I explained what the run would catch — the last three releases had each had at least one regression the gate caught, and one of them was a hallucination-rate increase that would have shipped to an enterprise customer under a compliance obligation. Then I offered options rather than a veto. Full run on the subset of the golden set that covered the changed surface, which was 80 cases instead of 500 and took six hours rather than three days. Ship one release behind the event on the standard gate. Or ship on the event date with the reduced gate plus a monitored canary at 10% with an on-call watch for four hours, accepting that we'd find problems in production with a fast rollback ready.

**R:** They took the third option. The canary caught nothing serious, but it did catch a 12% latency regression in the first hour that we fixed before full rollout — which is exactly what the gate exists for. The reduced gate became a standard pattern: evaluate the changed surface, not the whole set, for low-risk releases.

**L:** People don't want to skip quality, they want to skip *waiting*. If you can make the quality check cheap enough, the objection usually disappears. And giving three options including one that accepts risk explicitly is better than a veto, because it moves the decision back to the person who owns the trade-off instead of making me the obstacle.

---

## Section D — Influence without authority

### Q14. Tell me about a time you led a technical initiative without formal authority.

**What they're testing:** Whether you build consensus or escalate. The trap is a story where a manager told everyone to do what you said.

**S:** Four teams were each building their own LLM integration — separate prompt management, separate cost tracking, separate provider failover, four different ways of handling retries. Nobody owned the fragmentation because it had grown organically.

**T:** I had no authority over any of the teams. I wanted us to converge on a shared gateway layer.

**A:** I didn't start with a proposal. I started by measuring the duplication: I wrote a one-page comparison of what the four implementations did differently and cost each of them in maintenance attention. Two of the four had already been bitten by a provider outage and had no failover; one had no cost attribution at all and couldn't answer what their feature cost. Then I built the smallest useful version of the gateway myself — routing, failover, and token accounting — as a library, not a service, and put it behind a config flag in one team's system with their agreement. It took me a week and it didn't require anyone to adopt anything. When it survived a provider incident that would have taken them down, the other teams asked about it.

**R:** All four teams adopted within two months. The shared layer cut duplicated maintenance meaningfully, and the cost attribution immediately found one team whose feature was spending four times what they'd assumed because of a retry loop.

**L:** Adoption follows demonstrated value, not architecture diagrams. Building the smallest version and getting one willing adopter is faster than a design review that convinces everyone in principle and nobody in practice. And measuring the current pain in *their* units — maintenance hours, incidents — is what makes the case land.

### Q15. Describe a situation where you had to convince others to adopt a new approach.

**What they're testing:** Whether you understand the resistance or just push through it. The trap is framing the adopters as behind the times.

**S:** I wanted to introduce systematic evaluation — golden sets, judge scoring, CI gates — to a team that shipped on manual spot-checks and considered the eval work overhead.

**T:** I needed adoption from engineers who had no incentive to spend a week building an eval harness.

**A:** I found the one person on the team who had been burned by a silent quality regression — a prompt change that dropped faithfulness for a week before anyone noticed — and worked with them on their next change, building the eval set for their specific surface with them. When that change's gate caught a regression they'd have shipped, it was their story, not mine. I also made the first version deliberately small: 80 cases, not 500, and a run that took four minutes, because a gate that's slow gets disabled. And I stopped calling it evaluation in the pitch and called it "the thing that catches what you can't see," because the team's resistance was to ceremony, not to correctness.

**R:** Three of six engineers adopted within a quarter; within six months it was the default because the CI gate was catching roughly one real regression a month and the catches were visible in the release notes.

**L:** Adoption needs a champion who isn't you, and it needs the first version to be cheap enough that trying it isn't a commitment. I now look for the person with the scar and offer to build the thing with them, rather than pitching the team as a whole.

### Q16. Tell me about a time you influenced a decision you didn't own.

**What they're testing:** Whether you can be effective without being in charge. The trap is a story where you were right and the decision-maker eventually agreed because you kept pushing.

**S:** The team decided to use an off-the-shelf vector database with default index parameters for a 20-million-chunk corpus. I thought the defaults would fail at our scale and that we'd discover it in production.

**T:** I didn't own the decision. I had one design review to make the case.

**A:** Rather than arguing from principle, I ran the experiment. I loaded 2 million chunks with the default `ef_search` and measured recall against a known-relevant set, then compared it to a tuned configuration. Default recall@10 was 0.71; tuning to `ef_search=128` with `M=32` took it to 0.89 and added 40 milliseconds. Then I extrapolated with a caveat rather than a prediction — I said the degradation would likely worsen at 20 million because the graph gets sparser relative to the search breadth, and proposed a load test at 10 million before full migration rather than asking them to trust my extrapolation. I also gave the cheap option: tune the parameters in the existing system rather than changing databases, because that's what the data actually supported.

**R:** They tuned the parameters and ran the 10-million load test, which showed recall dropping to 0.68 at defaults. The full corpus shipped with tuned parameters and no production incident.

**L:** Influencing a technical decision works better with a measurement than an argument, and much better with a cheap option than a demand. The thing that made the difference wasn't my recall number — it was proposing a test they could run rather than a change they had to accept. It let them own the conclusion.

### Q17. Describe working with someone who had a very different working style.

**What they're testing:** Whether you adapt or judge. The trap is describing the other person's style as a deficiency.

**S:** I worked with a research scientist on productionising a retrieval method. I work in small increments with tests and ship continuously; they work in long focused stretches and consider a half-formed idea worth discussing before it's verifiable.

**T:** We had eight weeks to get their method into production with a 20% recall improvement claim to validate.

**A:** The friction was that I'd ask for a decision and they'd want to explore the space, and they'd propose a direction and I'd want the acceptance criteria. So we split the work by type rather than by phase. They owned the method's internals and the research questions, and I owned the production constraints and the validation. We agreed on the interface between us: they'd give me a version when it was stable enough to benchmark, and I'd give them the benchmark results with the latency budget attached, and neither of us would argue about the other's domain. When latency forced a simplification of their approach, I showed them the benchmarks proving it preserved their key innovation rather than just telling them it was necessary — and they contributed two further optimisations once they could see the constraint.

**R:** We shipped in eight weeks at 18% recall improvement — slightly below their 20% benchmark because of the latency constraint — and they published a follow-up on the production learnings. The interface agreement became the template for research-to-production collaboration on the team.

**L:** Most collaboration friction is an unstated interface, not a personality clash. Once we agreed on what each of us owned and what we'd hand over, the conflict disappeared, because the disagreements had been about boundaries rather than substance. I now set that up in the first week of any cross-disciplinary project.

---

## Section E — Disagreement and getting overruled

### Q18. Tell me about a disagreement with a colleague about a technical approach.

**What they're testing:** Whether you can separate the idea from the person. The trap is winning the story.

**S:** A colleague wanted to use a fine-tuned small model for a classification task; I argued for a prompted larger model with retrieval over historical examples. Both of us had reasonable positions and we were burning meeting time on it.

**T:** We needed to decide and neither of us would be persuaded by argument.

**A:** I proposed we stop arguing and settle it with a shared test. We agreed on the evaluation set together — 300 hand-labelled examples neither of us had built — and agreed in advance what would decide it: accuracy, cost per thousand, and latency, with a rule that if accuracy was within two points the other two decided. That last clause mattered, because it meant the decision wouldn't hinge on chasing small accuracy differences. We ran both, and the fine-tune won on accuracy by a point and a half and lost badly on cost at our volume, and the prompt approach won on the pre-agreed rule. My colleague was the one who pointed out that the fine-tune would likely win at ten times the volume and we should revisit it — which was a better conclusion than either of our opening positions.

**R:** We shipped the prompted version, and eight months later when volume had grown we revisited it and moved to the fine-tune, which was the right sequence.

**L:** Pre-agreeing the decision rule is what makes a technical disagreement resolvable in a day instead of a week. Arguing about which approach is better is unresolvable when "better" is undefined; arguing about a rule and then applying it is mechanical. And my colleague's observation about the volume threshold was only possible because we'd stopped defending positions.

### Q19. Describe a time you raised a concern and the team decided against you.

**What they're testing:** Whether you can lose gracefully and still add value. The trap is a story where the decision was reversed and you were vindicated.

**S:** Before a launch, I flagged that our agent's tool permissions were broader than the feature needed — it had write access to systems it only ever read from. Leadership weighed a two-week delay to scope the permissions against launch commitments and decided to ship as-is with a fast-follow ticket.

**T:** I disagreed with the call. My job was to make sure the decision was made with full information, then either escalate or commit.

**A:** I wrote the risk down concretely rather than repeating the general concern. I described the specific blast radius if the agent were prompt-injected — which systems could be written to, what an attacker could achieve, what the detection gap would be — as a realistic attack path, not a vague "this is risky." Then, instead of re-arguing the two-week delay, I asked for two mitigations that fit inside the launch window: an audit log on the write paths and an alert on anomalous write volume. Both were accepted in a day. I documented my disagreement in the decision record and committed to the launch without relitigating it in corridor conversations.

**R:** The launch went fine. Three weeks later the alert fired — on a misconfigured integration test, not an attack — but it proved the monitoring worked and it generated the evidence that got the fast-follow prioritised. Permission scoping shipped a month later.

**L:** Being overruled isn't the end of the job. Converting a lost argument into cheap guardrails plus a written record is usually worth more than winning, and the cheap guardrail is what produces the evidence that makes the next decision go the other way. It also builds the credibility that makes the next concern land harder — which is the actual long-term asset.

### Q20. How do you handle a teammate who's shipping AI work without evaluation?

**What they're testing:** Whether you handle it with the person or about them. The trap is escalating before talking.

**S:** A teammate was shipping prompt changes directly to production — reasonable changes, but with no test, no trace of which version was live, and no way to tell whether a change had helped or hurt. I found out because I was debugging a quality complaint and couldn't attribute it to any release.

**T:** I needed the practice to change without making it a performance issue.

**A:** I started with the problem I actually had, not with the practice I disapproved of. I told them I couldn't attribute a quality complaint because I couldn't tell which prompt was live, which made it my problem and a shared one. Then I made the fix trivially cheap: I added the prompt version to the trace field in about twenty minutes and showed them the query that would now answer "what changed and when." They adopted versioning for the trace reason immediately, and once the versions existed, the diff between two versions made the case for testing itself — they could see that the two changes they'd made that week had opposite effects.

**R:** Prompt versioning went in for the whole team within a month, and the teammate was the one who proposed the CI gate once they'd seen two versions side by side.

**L:** Lead with the problem you have rather than the practice they're missing. "I can't debug this" is unarguable and it's a problem you share; "you should be testing" is a criticism and it invites a defence. And making the first step twenty minutes of work is what turns a principle into a habit.

### Q21. Tell me about a time you had to give difficult feedback.

**What they're testing:** Whether you're specific and timely or vague and late. The trap is a story where the feedback was well-received because it was obviously correct.

**S:** A senior engineer on my team was writing evaluation sets that were, in practice, testing the system against cases it already passed. The scores were consistently high and the gates never fired. Their work looked good by every measure we had.

**T:** I needed to give feedback that would land, on a topic where the person's output looked successful.

**A:** I did it with the data, privately and quickly. I showed them the score distribution across their 400 cases — a spike at the top with almost no spread — and asked what a set that looked like that was capable of detecting. They got there themselves in about a minute, which was the point; my job was to show the distribution, not deliver the verdict. Then we worked on the fix together: how to sample from production failures rather than from representative queries, why you stratify by difficulty rather than sampling randomly, and why you keep a slice you don't look at. I made it a shared problem by saying I'd made the same mistake early on, which was true.

**R:** Their next set had a real spread and immediately surfaced two regressions that the previous set had been blind to. They subsequently became the person who reviewed other people's eval sets.

**L:** Feedback lands when the person reaches the conclusion themselves from evidence you provide. Showing the distribution rather than stating the flaw did that. And the "I made this mistake too" isn't a softening tactic — it's true for almost every eval mistake, because saturated eval sets are the default failure and nearly everyone builds one first.

---

## Section F — Responsible AI and safety

### Q22. Tell me about a time you raised an ethical concern about an AI system.

**What they're testing:** Whether ethics is a real practice for you or a talking point. The trap is a story with no cost to you.

**S:** We were building a resume screening system to help recruiters process high volumes of applications. During development I noticed the training data was heavily skewed toward engineers from a small set of universities.

**T:** I needed to raise it in a way that would be taken seriously rather than filed as "slowing the project down."

**A:** I quantified it before I said anything. The model had 80% precision on candidates from top-tier universities and 45% on candidates from state schools with comparable qualifications — a gap that was invisible in the aggregate accuracy number because the aggregate was fine. I presented it not as "we shouldn't do this" but as a legal and business exposure, with the specific regulatory context and the recent enforcement actions in hiring. Then I proposed two options with different trade-offs: rebalance the training data for diverse representation, or use the model only for matching candidates to roles and never for ranking them, with human review of every candidate. The team took the first option plus adding fairness metrics as a standing part of our evaluation suite.

**R:** We delayed launch by three weeks and shipped a system whose performance was consistent across demographic groups. The fairness metrics became standard for our models, and legal cited the proactive work in a customer security review.

**L:** Framing an ethical concern as a business and legal risk makes it actionable for people who don't respond to the ethical framing. That's not cynicism — it's translation, and translation is what gets the outcome. And raising it during development rather than after launch was the whole difference: the cost of the fix was three weeks, and the cost of the alternative would have been a lawsuit.

### Q23. How do you think about fairness in AI systems?

**What they're testing:** Whether you have a working definition or a slogan. The trap is a recitation of principles with no operational detail.

**S:** This is a question rather than a story, so I answer it with the operational version.

**T:** — how I actually apply it.

**A:** I start from the fact that you cannot measure fairness without picking a definition, and the definitions disagree with each other — demographic parity, equalised odds, and individual fairness are mathematically incompatible in most real settings, so "make it fair" is not a specification. So my practice is: define the protected groups that matter for this decision, define the harm you're trying to avoid, choose the metric that matches that harm, and write down what you're trading off by choosing it. Then measure it as a slice metric in the evaluation suite, not as a one-time audit, because fairness drifts as the data drifts. In practice the two things that move the needle most are the data — representation in training and in the eval set — and the decision boundary, because a model used to *rank* creates different harms than the same model used to *filter*.

**R:** The concrete version of this that I've shipped: fairness metrics in the CI gate alongside accuracy, so a change that improves the aggregate and degrades a protected slice fails the build the same way a latency regression would.

**L:** Fairness only becomes real when it's a number in a gate that can fail. Principles in a document don't change behaviour; a red build does. The hard part isn't agreeing that fairness matters — it's choosing which metric and being honest about what that choice gives up.

### Q24. Describe a situation where you prioritised safety over speed.

**What they're testing:** Whether safety is a real constraint you've paid for. The trap is a story where safety cost nothing.

**S:** We were two weeks from shipping an agent that could issue refunds up to $500, and it was performing well — 94% of its refund decisions matched what a human would have done in our offline evaluation.

**T:** I owned the action-safety design and I recommended we not ship autonomous refund issuance.

**A:** The 94% number was the problem, not the reassurance. The failures weren't random — they were concentrated on partial refunds and subscription proration, which are exactly the cases where the amount is ambiguous. And the failure was a $500 irreversible transfer, not a wrong sentence. I proposed shipping with a confirmation gate instead of autonomy: the agent prepares the refund, shows the amount and the reasoning, and the human clicks once. That cost us the headline feature — "autonomous refunds" became "assisted refunds" — and it added about four seconds to the interaction. I made the case with the failure categorisation rather than with risk in the abstract, and I committed to revisiting autonomy for the refund types where the agent was 99%+ accurate.

**R:** We shipped assisted, and over the next quarter the refund types with 99%+ accuracy turned out to be about a third of the volume — we enabled autonomy for those, and kept the gate for the ambiguous remainder. Net effect: most of the value, none of the irreversible mistakes.

**L:** Safety is genuinely slower in the first release and usually faster overall, because the conservative version generates the accuracy data that tells you where autonomy is safe. The question I'd ask now is not "is the agent good enough" but "what's the blast radius of the failure and is it reversible" — those give different answers, and the second one is the one that matters.

### Q25. How do you handle a stakeholder who wants to cut safety measures?

**What they're testing:** Whether you can hold a line without being rigid, and whether you distinguish the control from the implementation.

**S:** A business owner asked me to disable the output PII filter because it was blocking about 4% of legitimate responses — a customer's own data being quoted back to them was being flagged and blocked.

**T:** I owned the guardrail configuration.

**A:** The 4% was a real problem and I said so — a filter that blocks legitimate answers is a product defect, not a cost of doing business. But the request as framed was to turn it off, and that trades a 4% friction problem for a PII egress problem, which is a reportable event under our obligations. So I separated the control from the implementation. The control is "PII does not leave the system to an unauthorised recipient." The implementation was a naive pattern matcher that couldn't tell the user's own data from someone else's. I proposed fixing the implementation instead: keep the filter, but add an exception for data whose provenance trace showed it came from the requesting user's own authorised context. That required a provenance field we partly had.

**R:** Two weeks of work got the false-positive rate from 4% to 0.6% without weakening the actual control. The provenance field also improved several other things — we could finally answer which documents a response had drawn from.

**L:** Most requests to remove a control are actually requests to fix a bad implementation of it. Separating the two turns an argument about policy into an engineering problem with a solution, and it means you're never in the position of defending a control that's genuinely harming users. The one thing I'd never do is argue that the 4% doesn't matter, because it does — I just don't accept that the fix is removal.

---

## Section G — Growth, craft and judgement

### Q26. How do you stay current with the rapidly changing AI field?

**What they're testing:** Whether you have a filtering practice or you read everything and retain nothing. The trap is listing sources rather than describing how you decide what matters.

**S:** A question rather than a story.

**T:** — my practice, concretely.

**A:** Three habits. I read primary sources rather than summarised takes for anything I might act on — the model card, the paper, the provider's changelog — because summaries lose exactly the caveat that changes the decision. I filter by **whether it changes a decision I'm currently making**: a new model that's 3% better on a benchmark doesn't change anything, but a pricing change or a context-limit change or a new caching behaviour changes my architecture, and those I act on within days. And I keep a small number of things I build with — one or two side projects — because reading about a technique and hitting its failure modes are different kinds of knowledge and only the second is transferable.

The filtering rule I'd articulate: most new capability announcements don't change a design decision, and treating them as if they do is how teams churn their stack. The ones that matter are changes to the cost curve, the latency floor, the context limit, or the reliability characteristics — those change architecture. Benchmark deltas change model selection at the margin.

**R:** The concrete outcome is that we've changed our model pin three times in eighteen months and our architecture once, and both changes came from a cost or capability shift rather than from a leaderboard.

**L:** The scarce skill isn't finding new information, it's deciding what not to act on. I now ask "does this change a decision I have open" before spending attention, and the answer is usually no.

### Q27. Tell me about a skill you developed recently.

**What they're testing:** Whether you learn from need or from interest, and whether the learning was applied. The trap is a course you took.

**S:** We had a reliability problem I couldn't diagnose: intermittent quality complaints that I couldn't attribute to any release, and no mechanism to find out what was different about the failing requests.

**T:** I needed to learn enough about production observability for LLM systems to build the instrumentation myself, because nobody on the team had done it.

**A:** I learned it by building the smallest useful version rather than by reading first. I added span attributes to the request path — model pin, prompt version, index snapshot, retrieved chunk IDs with scores, token counts, and cache hit status — and started logging them. Then I wrote a script that grouped failing requests by each attribute to see which one segregated them. The first version was crude and the answer came out of it anyway: the failures clustered on requests where the semantic cache had hit, which pointed straight at a false-hit problem. Once I had that, I went back and read properly about the standard tracing schemas and judge calibration, because then I knew which parts I needed.

**R:** The instrumentation found a 2% cache false-hit rate that had been invisible, and the same trace schema is now the base for our eval sampling, cost attribution, and audit records. Three distinct problems got solved by one schema.

**L:** Learning by building a rough version first is much faster than learning thoroughly and then building, because the rough version tells you which twenty percent of the field you actually need. I now deliberately build the crude thing before I read the literature, which is the opposite of what I did earlier in my career and about three times faster.

### Q28. What's the biggest mistake you've made as an engineer?

**What they're testing:** Whether the mistake has a structural lesson or is a war story. The trap is a mistake that's actually a humblebrag.

**S:** Early in my work on RAG systems, I tuned retrieval extensively — chunking strategy, embedding model, hybrid weights, reranker — over about six weeks, and the system's measured quality barely moved. I was convinced the retrieval was the bottleneck.

**T:** I owned retrieval quality and I reported that I was making no progress.

**A:** A colleague asked a question I should have asked myself on day one: when the system gives a bad answer, do the retrieved chunks actually contain the answer? I went back through 200 failed cases and labelled them. In 71% of them, the correct information *was* in the retrieved context and the model had produced a wrong or unfaithful answer anyway. My retrieval changes had been improving a stage that was already good enough, and I'd spent six weeks on it because retrieval was the part I knew how to tune. The actual problem was generation — a prompt that didn't make clear it should answer only from the provided context and say so when the context was insufficient — which took three days to fix.

**R:** Faithfulness went from 0.71 to 0.88 and the hallucination rate from 9% to 2.5% in under a week. More importantly, I built the failure-attribution step — labelling failures as retrieval-miss, retrieval-precision, or generation — into the diagnostic process permanently.

**L:** I was optimising the thing I knew how to measure rather than the thing that was broken. The fix wasn't technical, it was procedural: attribute the failure to a stage before tuning that stage. I now refuse to optimise any component until I can point at the failures it's responsible for, and that one habit has been worth more than any specific technique I've learned since.

---

## Questions to ask your interviewers

Strong questions signal that you evaluate systems the way someone who has operated them does. Ask the ones whose answers would actually change your decision.

**About the work**

- How do you currently decide whether a model or prompt change was an improvement? What would you change about that?
- What's the biggest technical challenge the team is facing right now, and is it a capability problem or an operational one?
- What does your release process look like when a prompt changes — is it a deploy, a config change, or something else?

**About failure**

- Can you tell me about the last quality regression you had in production, and how you found out about it?
- How does the team handle a model provider changing behaviour without notice?

**About responsibility**

- What's the process for raising a safety or ethics concern, and can you give me an example of it changing a decision?
- How does the team balance shipping speed against evaluation rigor, and where has that been tested?

**About growth and leveling** (save for later-stage conversations)

- What does success in this role look like at six months versus a year?
- What separates this level from the next one here — can you give an example of someone who made that jump?
- How is compensation structured at this level: base, bonus, equity split, and refresh cadence?
- Is there a separate track or premium for AI-critical skills, and how is it reviewed as the market moves?

---

## Red flags to avoid

| Behavior | Why it reads badly | The fix |
|---|---|---|
| Blaming others | You'll do it about them next | Use "I" for actions and "we" for outcomes |
| No specific examples | Suggests inflated experience | One number per story makes it unfakeable |
| Only technical detail | Misses that this job is mostly coordination | Add the stakeholder and the trade-off |
| Dismissing ethics or safety questions | Signals future liability | Have one real story where safety cost you something |
| All victories | Nobody has only wins; it reads as curation | One wrong-and-walked-it-back story is required |
| No questions for the interviewer | Reads as low engagement | Prepare five, ask three |
| Rambling past three minutes | Poor signal on communication under constraint | Rehearse the 30-second version too |
| "We" for everything | Hides your contribution | Be specific about what you personally decided |

---

## Preparation checklist

- [ ] Prepare 6–8 stories covering: ambiguity, failure, conflict, influence, ethics, and learning
- [ ] Include at least one where you were **wrong and walked it back**, and one where you were **overruled and committed anyway**
- [ ] Give every story a **number** in the Result and a **changed default** in the Learning
- [ ] Include at least two explicitly AI-specific ones: a silent quality regression, and a safety-versus-speed trade
- [ ] Time each story at 2 minutes, and rehearse a 30-second version of each for "tell me briefly"
- [ ] Record each story once and listen for filler, buried results, and unrequested context
- [ ] Do one mock where the interviewer interrupts with "why did you do that?" and "what would you do differently?"
- [ ] Prepare 3–5 interviewer questions, plus the leveling and compensation set for late-stage calls
- [ ] Re-read the red-flag table the night before and check your draft stories against it

---

## Cross-links

- [LLM, RAG and GenAI Questions](04-llm-rag-genai-questions.md) — the technical bank behind these stories
- [Agentic AI Questions](05-agentic-ai-questions.md)
- [LLMOps and Operationalisation Questions](06-llmops-and-operationalisation-questions.md)
- [Foundations and Capacity Questions](01-foundations-and-capacity-questions.md)
- [Production RAG System](../HLD/05-ai-systems/01-production-rag-system.md) — the failure modes behind Section B
- [Reliability Patterns](../Operations/07-reliability-patterns.md) — silent quality regression in depth
- [Security, Guardrails and Compliance](../Operations/08-security-guardrails-and-compliance.md) — the blast-radius framing in Section F
- [LLMOps for AI Systems](../Operations/10-llmops-for-ai-systems.md) — attribution and the four causes of a quality drop
