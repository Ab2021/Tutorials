# Day 94: Creative Writing & Content Agents
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you prevent "LLM-ese" (repetitive, bland style)?

**Answer:**
*   **Temperature:** Increase to 0.8 for creativity.
*   **Frequency Penalty:** Increase to prevent word repetition.
*   **Negative Prompting:** "Do not use the words: delve, tapestry, landscape, leverage."
*   **Few-Shot:** Show examples of "Good" (Human) vs "Bad" (AI) writing.

#### Q2: How do you handle Factuality in creative writing?

**Answer:**
*   **RAG:** Retrieve facts first, then write.
*   **Citation:** Force the model to link every claim to a source.
*   **Fact-Check Step:** A separate agent verifies claims against Google *after* generation.

#### Q3: What is "Brand Voice Alignment"?

**Answer:**
Ensuring the AI sounds like the company.
*   **Brand Bible:** A document defining the persona (e.g., "We are helpful, authoritative, but not arrogant").
*   **Fine-Tuning:** The most robust way. Fine-tune on the company's past 1,000 blog posts.

#### Q4: How do you optimize for SEO without ruining readability?

**Answer:**
*   **Keyword Density:** Instruct the model to use keywords naturally.
*   **Semantic SEO:** Focus on answering "People Also Ask" questions rather than keyword stuffing.
*   **Structure:** Use H2/H3 tags effectively.

### Production Challenges

#### Challenge 1: Length Drift

**Scenario:** You ask for 500 words. Model gives 200.
**Root Cause:** LLMs are lazy.
**Solution:**
*   **Sectioning:** Ask for 5 sections of 100 words each.
*   **Min Tokens:** Use `min_tokens` parameter (if supported) or reject short outputs.

#### Challenge 2: Context Window Limits (Books)

**Scenario:** Writing a novel. Model forgets Chapter 1.
**Root Cause:** Context limit.
**Solution:**
*   **Rolling Summary:** Maintain a 500-word summary of the story so far.
*   **Character Bible:** Inject character sheets (Name, Traits) into every prompt.

#### Challenge 3: Copyright & Plagiarism

**Scenario:** Model outputs text identical to a copyrighted article.
**Root Cause:** Overfitting.
**Solution:**
*   **Plagiarism Checker:** Run output through Copyscape/Grammarly API before publishing.
*   **Paraphrasing:** If match found, ask the model to "Rewrite this paragraph completely."

### System Design Scenario: Automated Newsroom

**Requirement:** Generate 100 news articles per day from press releases.
**Design:**
1.  **Ingest:** Monitor RSS feeds.
2.  **Filter:** Classify relevance.
3.  **Draft:** Agent A writes the story.
4.  **Headline:** Agent B generates 10 headlines. Agent C picks the best (CTR prediction).
5.  **Image:** Agent D generates a DALL-E prompt -> Image.
6.  **Review:** Human editor approves.

### Summary Checklist for Production
*   [ ] **Human Review:** Never publish without human eyes (for brand safety).
*   [ ] **Disclosure:** Tag AI content if required by law.
*   [ ] **Feedback Loop:** If an editor changes the AI draft, save the diff to fine-tune the model later.
