### Interview pack

Fourteen questions, mixed across recall, calculation, diagnosis, and judgment. Nine require computation or a design decision, five test whether the candidate can reject a plausible-but-wrong framing. Score against the rubric in each item; a candidate who reaches the right number by the wrong route should be scored on the route, because the route is what transfers.

#### Q1. A promotion delivers a 60% sales lift and the break-even requirement is 80%. What is your recommendation, and what else do you need before stating it?

**Answer.** It fails the payback test on the stated terms, so the default recommendation is not to repeat at that depth. But before stating it, I need the realised depth (not the advertised one), the funding split, and the window. Each can move the requirement: 50% supplier funding drops the 80% requirement to 28.57%, at which point the same 60% lift clears the bar on the item; the full window including carryover can then push it back below.

**Reasoning.** The requirement is (m_r − m_p)/m_p, so it is a function of margin, not of the lift. Reporting "60% versus 80%" as a verdict treats the funding terms as fixed background when they are the largest single lever on the threshold.

**Follow-up.** If I told you the supplier funds half the discount and the item is shelf-stable with a two-week consumption cycle, what would you predict about the full-window number? (Expected: negative, because the promoted margin is smaller than the displaced regular margin, so displacement below 100% flips the sign.)

**Red flag.** Answering "60% is close to 80%, so it's roughly break-even". Close is not a category; the requirement is a point value and the estimate has an interval, and neither supports "roughly".

**Scoring.** 3 = rejects on the stated terms, names funding and window as the two levers, predicts the carryover sign. 2 = rejects but treats the requirement as fixed. 1 = reports the lift as a result.

#### Q2. Compute the break-even volume for a 25% price cut on an item priced at $4.00 with a landed cost of $2.20. Then compute it if the supplier funds 60% of the discount.

**Answer.** Unfunded: p_p = $3.00, m_p = $0.80, m_r = $1.80, so the requirement is (1.80 − 0.80)/0.80 = 125.0% more volume. Funded at 60%: f = 0.6 × $1.00 = $0.60, m_p = $1.40, requirement = 0.40/1.40 = 28.57%.

**Reasoning.** The requirement is the margin given up divided by the margin retained. Funding raises the retained margin, so it lowers the numerator and raises the denominator simultaneously — a double effect, which is why a 60% funding share cuts the requirement by 77% (125.0 → 28.57) for the same 25% cut.

```text
CHECK: m_r = 4.00 - 2.20 = 1.80 ; p_p = 4.00*0.75 = 3.00 ; m_p = 3.00 - 2.20 = 0.80
       (1.80 - 0.80)/0.80 = 1.000/0.80 = 1.25 -> 125.0%
       f = 0.60*(4.00 - 3.00) = 0.60 ; m_p' = 0.80 + 0.60 = 1.40
       (1.80 - 1.40)/1.40 = 0.40/1.40 = 0.285714 -> 28.57%
       ratio 28.57/125.0 = 0.2286, so a 77.1% reduction in the requirement
```

**Follow-up.** At what cost would the unfunded 25% cut be profitable at a 60% observed lift? (Solve 0.60 = (m_r − m_p)/m_p with p_p = 3.00: m_p = m_r/1.60, and m_r = m_p + 1.00, so m_p + 1.00 = 1.60m_p → m_p = 1.667, c = 1.333. A cost of $1.33 rather than $2.20 — a 39% lower cost basis.)

**Red flag.** Computing the requirement as "discount divided by price", i.e. 25%, which ignores that the cost does not move.

**Scoring.** 3 = both numbers correct with the funding's double effect explained. 2 = correct unfunded, arithmetic slip on the funded case. 1 = uses discount/price as the hurdle.

#### Q3. Your comparison stores' sales rose 35% during the event while treated stores rose 60%. What is the attributable lift, and what would you check first?

**Answer.** k = 1.35, so the counterfactual is 1.35 × baseline and the attributable lift is 1.60/1.35 − 1 = 18.52%, not 25% (which is the naive subtraction 60% − 35%). The first check is whether the comparison stores were actually untreated: a 35% movement in a non-promoted group is either a national shock or contamination.

**Reasoning.** Division, not subtraction, because the effects are multiplicative on a level. 1.60/1.35 = 1.1852. The subtraction error (25%) overstates the effect by 35% relative to the correct figure, and the error grows with k — at k = 1.50 the correct answer is 6.67% against a subtracted 10%.

**Follow-up.** If the comparison group turns out to be contaminated by a regional circular, what is the cheapest valid design left? (Reclassify as treated and search for donors outside the zone; if none, state that the event is not causally evaluable and fall back to a within-item depth comparison across events.)

**Red flag.** Subtracting percentages. Also: assuming the 35% is a national shock without checking the price file.

**Scoring.** 3 = 18.52% and names contamination as the first check. 2 = 18.52% without the diagnostic. 1 = 25%.

#### Q4. Explain why a promotion with a 12% sales lift can be more profitable than one with a 40% lift, using the numbers you would need.

**Answer.** Profit is dQ·m_p − Q_cf·(m_r − m_p), so a small lift on a high-margin, shallow-discount item can beat a large lift on a low-margin, deep-discount item. Concretely: item A lifts 12% at a 5% cut with m_r = $2.00 and m_p = $1.85, giving 0.12 × 1.85 − (1.00 × 0.15) = +$0.072 per counterfactual unit. Item B lifts 40% at a 25% cut with m_r = $1.80 and m_p = $0.80, giving 0.40 × 0.80 − 1.00 × (1.80 − 0.80) = 0.32 − 1.00 = −$0.68 per counterfactual unit. A earns +$0.072 per counterfactual unit at 12% lift; B loses $0.68 at 40% lift.

**Reasoning.** The two terms scale differently: the gain term is proportional to dQ × m_p while the loss term is proportional to Q_cf × (m_r − m_p), which is independent of dQ. Small lifts on shallow cuts win because the loss term is small, and the requirement (m_r − m_p)/m_p is the exact ratio at which the two terms balance.

**Follow-up.** What would you need beyond these numbers to make the comparison decision-grade? (Substitution sets for both items, the funding split, and the carryover rate — because B's deep cut on a storable item will have a much larger displacement rate than A's shallow cut.)

**Red flag.** Saying "it depends on margin" without producing the arithmetic that shows how.

**Scoring.** 3 = writes the two-term identity and prices both items correctly. 2 = qualitative answer with correct direction. 1 = claims higher lift always wins.

#### Q5. What is the difference between sell-in and sell-out, and which one belongs in an uplift calculation?

**Answer.** Sell-in is the retailer's purchase from the supplier; sell-out is the shopper's purchase at the register. Uplift belongs on sell-out, because the decision is about consumer demand. Sell-in belongs in a working-capital discussion, because an off-invoice deal makes forward buying rational and produces sell-in that exceeds sell-out with no incremental demand at all.

**Reasoning.** If the numerator is sell-in, the measured "effect" includes inventory accumulation at a discounted cost, which is a financing gain rather than a demand gain. It also double counts: the forward-bought units are sold at regular price later, so their margin appears once in the event period and again in the following period, and neither appearance is incremental demand.

**Follow-up.** How would you distinguish retailer forward buying from household stockpiling using one loyalty extract? (Household repeat frequency rises during the event and falls after under stockpiling; it stays flat under retailer forward buying, while the retailer's inventory position rises.)

**Red flag.** "They're both sales, so it doesn't matter" — or using sell-in because it is the only feed with a clean join to the funding data.

**Scoring.** 3 = defines both, picks sell-out for the decision, and gives the loyalty discriminator. 2 = picks sell-out without the discriminator. 1 = uses sell-in.

#### Q6. Your uplift estimate is +8% with a 95% interval of [1%, 15%] and the pre-registered payback requirement is 12%. What is the decision?

**Answer.** Defer with a named test. The interval excludes zero, so there is evidence of a positive effect, but it contains 12%, so the decision is genuinely unresolved. "Defer" is one of the six options and must carry an owner, a date, and the specific evidence that would resolve it — here, either a larger estate to narrow the interval or a pooled test across items with the same depth and funding structure.

**Reasoning.** Reporting "significant, therefore repeat" converts a probability statement into a profit claim. The interval's upper end is 15%, which at a 12% requirement means the event could be worth repeating; the lower end is 1%, which means it could be negligible. The expected value may even be positive, but the variance around the threshold is what the decision has to tolerate.

**Follow-up.** What would you do differently if the requirement were 5% instead of 12%? (Repeat: the whole interval is above the requirement, so the decision is unambiguous even though the effect is small.)

**Red flag.** "It's significant, so it works." Also: widening the interval by inventing uncertainty rather than reporting the computed one.

**Scoring.** 3 = defer with the exact resolving evidence named. 2 = defer without a resolution path. 1 = repeat, or stop.

#### Q7. You are asked to evaluate 40 promotions with 4 analyst-days per week. What do you change about the method?

**Answer.** Tier the portfolio. Evaluate the top 8 events by spend with the full design (matched DiD, substitution, carryover, precision report); evaluate the middle 20 with a within-item depth comparison across historical events, which needs no new data collection; and report the bottom 12 descriptively with no causal claim. Publish the tiering rule in advance so it is not a popularity contest.

**Reasoning.** A method that cannot be executed weekly is bypassed, and a bypassed method produces no evidence at all — worse than a crude method that runs. The tiering also concentrates effort where the decision value is highest: an event with $400k of discount at stake justifies a day of analysis; one with $4k does not.

**Follow-up.** What is the failure mode of tiering? (The untested tail drifts: events that were never evaluated get repeated because nobody has evidence against them, so the tail grows. Mitigation: a mandatory retirement rule — an event unevaluated for three cycles drops to the lowest depth by default.)

**Red flag.** Proposing to evaluate all 40 properly, which is a plan that will fail in week two and take the credibility of the whole programme with it.

**Scoring.** 3 = tiers, states the rule in advance, names the tail failure. 2 = tiers without the failure mode. 1 = promises all 40.

#### Q8. A colleague reports that the promotion "paid for itself" because incremental revenue of $1,536 exceeded the discount given of $1,280. What is wrong with that?

**Answer.** Revenue is not profit. The discount is a transfer out of margin, and the incremental revenue arrives at the promoted margin, not at the full price. The incremental units were 480 at a promoted margin of $1.00, contributing $480 of margin, against a loss of $896 on the 1,120 units that would have sold at $1.80 and instead earned $1.00. Net −$416. The revenue-minus-discount comparison nets a gross inflow against a margin outflow, so it can be positive while the event loses money.

**Reasoning.** The break-even condition compares margin on incremental units against margin lost on the units that would have sold anyway. In this example the loss term is Q_cf × $0.80 = $896, which is larger than the entire incremental margin of $480. Matching revenue to discount matches a gross figure to a net figure, which is a dimensional error dressed as an accounting one.

```text
CHECK: incremental revenue = 480 * 3.20 = 1,536.00 ; discount given = 1,600 * 0.80 = 1,280.00
       incremental margin  = 480 * 1.00 = 480.00 ; margin lost = 1,120 * 0.80 = 896.00
       net = 480 - 896 = -416.00
       the "revenue equals discount" claim compares 1,536 to 1,280 and calls the
       256 difference profit, which is revenue minus a margin cost, not profit
```

**Follow-up.** Construct a case where revenue really would equal discount as a break-even condition. (Only if the cost were zero: then m_p = p_p and m_r = p_r, and the break-even condition reduces to dQ·p_p = Q_cf·(p_r − p_p), i.e. incremental revenue equals discount. A zero-cost item is the special case where the colleague's rule is exactly right.)

**Red flag.** Accepting the framing and arguing about the size of the numbers instead of the dimension of the comparison.

**Scoring.** 3 = names the revenue-versus-margin error and reconstructs the two-term identity. 2 = says "revenue isn't profit" without the arithmetic. 1 = agrees.
