# 🛍️ GenAI Product Formulations in E-Commerce (Flipkart)
### Flipkart Senior Data Scientist — Interview Preparation

> **Context:** E-commerce GenAI is moving beyond simple "chatbots." At Flipkart, GenAI is used to fundamentally alter product discovery, automate catalog operations, and reduce massive operational costs (like RTOs and returns). This document outlines three core GenAI e-commerce products formatted for product sense interviews.

---

## 1. The Conversational Shopping Assistant
**The Problem:** Traditional search is keyword-based ("Red running shoes size 10"). But users often shop by *intent* or *occasion* ("I need an outfit for a summer wedding in Goa under ₹5000"). Keyword search fails completely on intent-based queries.

**The Product Vision:** A multi-modal GenAI agent that understands complex, multi-turn natural language queries, asks clarifying questions, and curates a specific list of products dynamically.

### Product Formulation (The GAME Framework)
*   **Goal:** Increase conversion rate for high-intent, complex queries and reduce the bounce rate on the search page.
*   **Audience:** High-intent shoppers who don't know exactly what product SKU they want (Discovery phase).
*   **Metrics:**
    *   *North Star:* **Session-to-Cart Conversion Rate** for users who interact with the assistant vs. standard search.
    *   *Engagement Metric:* Multi-turn retention (How many users reply to the bot's follow-up question vs. dropping off).
    *   *Counter Metric:* Cannibalization of standard search revenue (Are we just shifting users who would have bought anyway, or are we creating incremental GMV?).
*   **Execution & Trade-offs:**
    *   *Architecture:* LangChain Tool-Calling Agent. The LLM doesn't have the catalog memorized. It acts as a router that writes GraphQL/SQL queries to the Flipkart catalog API (e.g., `Tool: search_catalog(category="ethnic wear", max_price=5000, tags=["summer", "wedding"])`).
    *   *The Trade-off:* Latency vs. Quality. An LLM agent loop (Thought -> Action -> Observation) takes 3-5 seconds. E-commerce users expect <500ms search results. We must stream intermediate thoughts to the UI ("Searching the catalog for summer ethnic wear...") to prevent the user from bouncing while waiting.

---

## 2. GenAI Catalog Enrichment (Automated SEO & Descriptions)
**The Problem:** Flipkart has millions of sellers. Many upload products with terrible titles ("Shirt blue XL") and blank descriptions. This destroys SEO, makes keyword search fail, and lowers user trust. Manually writing descriptions for 100 million SKUs is impossible.

**The Product Vision:** A backend multimodal LLM pipeline that looks at the seller's uploaded image and basic specs, and automatically generates high-converting, SEO-optimized product titles, bullet points, and rich descriptions.

### Product Formulation
*   **Goal:** Increase organic search traffic (SEO) and improve product page conversion rates (CVR) by building consumer trust through rich content.
*   **Audience:** Internal Catalog Team & Sellers.
*   **Metrics:**
    *   *North Star:* **Organic Search Traffic Lift** for products that received the GenAI enrichment vs. a control group.
    *   *Quality Metric:* **Edit Distance**. A random sample of AI descriptions is sent to human copywriters. We measure how much they have to edit the AI's draft. If Edit Distance is near zero, the model is production-ready.
    *   *Safety Metric:* Hallucination rate. (e.g., The AI sees a blue shirt and hallucinates that it is '100% Egyptian Cotton' when the fabric isn't specified).
*   **Execution & Trade-offs:**
    *   *Architecture:* Multimodal pipeline (e.g., GPT-4 Vision or Gemini 1.5 Pro). Input: `[Product Image] + [Raw Seller Data]`. Output structured JSON: `{title: "...", bullet_points: [...], description: "..."}`.
    *   *The Trade-off:* Hallucination vs. Richness. If you tell the LLM to write a compelling description, it will invent features to sound good. You must use constrained generation: "Only write about features explicitly visible in the image or stated in the specs." We sacrifice marketing "fluff" for legal safety.

---

## 3. Post-Purchase Troubleshooting Agent (Return Deflection)
**The Problem:** Electronics have high return rates. Often, the product isn't broken; the user just doesn't know how to set it up (e.g., "My Bluetooth headphones won't pair"). Returns cost Flipkart massive logistics fees and inventory depreciation.

**The Product Vision:** When a user clicks "Return Item" for an electronic product, a GenAI troubleshooting bot intercepts them. It has ingested the exact product manual for that specific SKU and attempts to solve the user's problem via a chat interface before allowing the return.

### Product Formulation
*   **Goal:** Reduce the overall Return Rate for electronics by solving user-error issues.
*   **Audience:** Post-purchase customers experiencing friction.
*   **Metrics:**
    *   *North Star:* **Return Deflection Rate**. The percentage of users who clicked "Return Item", interacted with the bot, and subsequently *cancelled* their return request.
    *   *Customer Metric:* Net Promoter Score (NPS) or CSAT of the interaction.
    *   *Counter Metric:* Customer Support Escalations. If the bot is frustrating, they will just call the human support line angry.
*   **Execution & Trade-offs:**
    *   *Architecture:* Classic RAG. We index the PDF manuals of the top 10,000 electronics SKUs into a Vector Store. When the user asks "How do I pair this?", the bot retrieves the pairing chunk from the exact manual of the SKU they purchased.
    *   *The Trade-off:* Friction vs. CX (Customer Experience). If we force every user to talk to a bot before returning a truly broken item, they will hate the Flipkart experience and buy from Amazon next time. We must provide an "Escalate to Return" button immediately, making the bot an *opt-in* helper rather than a frustrating roadblock.
