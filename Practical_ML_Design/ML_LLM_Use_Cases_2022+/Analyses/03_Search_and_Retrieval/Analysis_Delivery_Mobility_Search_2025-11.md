# ML Use Case Analysis: Search & Retrieval in Delivery & Mobility Industry

**Analysis Date**: November 2025  
**Category**: Search & Retrieval  
**Industry**: Delivery & Mobility (Food Delivery, Grocery, Ride-Sharing)  
**Articles Analyzed**: 4 (DoorDash, Instacart, Foodpanda, Uber)

---

## PART 1: USE CASE OVERVIEW

### 1.1 Basic Information

**Category**: Search & Retrieval  
**Industry**: Delivery & Mobility  
**Companies**: DoorDash, Instacart, Foodpanda, Uber Eats  
**Years**: 2022-2025  
**Tags**: Real-Time Search, Geo-Spatial Retrieval, Knowledge Graphs, Substitution Logic, Hybrid Retrieval

**Use Cases Analyzed**:
1. [DoorDash - Leveraging LLMs for Better Search Retrieval](https://doordash.engineering/2024/04/02/leveraging-llms-for-better-search-retrieval/) (2024)
2. [DoorDash - Building a Product Knowledge Graph](https://doordash.engineering/2024/02/27/building-a-product-knowledge-graph-with-llms/) (2024)
3. [Instacart - Optimizing Search Relevance with Hybrid Retrieval](https://tech.instacart.com/optimizing-search-relevance-at-instacart-using-hybrid-retrieval) (2024)
4. [Foodpanda - Classifying Restaurant Cuisines](https://medium.com/foodpanda-tech/classifying-restaurant-cuisines-with-subjective-labels) (2022)

### 1.2 Problem Statement

**What business problem are they solving?**

The Delivery industry faces a **"Hyper-Local, Real-Time"** search problem that is fundamentally different from web search or standard e-commerce.

- **DoorDash/Uber Eats**: "Thai food" isn't a global query. It means "Thai restaurants *currently open*, *within 5 miles*, that *can deliver in <45 mins*".
- **Instacart**: "Organic Strawberries" isn't just a product match. It's a check for inventory at the specific store branch the user is shopping from. If out of stock, what is the best *substitute*?
- **Foodpanda**: "Spicy Noodles" might be a dish name, a category, or a description. Mapping unstructured menu text to structured cuisine tags is critical.

**What makes this problem ML-worthy?**

1.  **Inventory Volatility**: Products go out of stock in minutes. The search index must reflect real-time reality.
2.  **Geographic Constraints**: Retrieval is strictly bounded by user location. A perfect match 10 miles away is a bad result.
3.  **Substitution Logic**: If "Brand A Milk" is missing, "Brand B Milk" is a good sub, but "Brand A Orange Juice" is not. This requires deep semantic understanding of product attributes.
4.  **Menu Unstructured Data**: Millions of small restaurants upload menus with typos, vague descriptions ("Delicious Combo"), and no standard taxonomy.
5.  **Cold Start**: New restaurants/products appear daily with no click history.

---

## PART 2: SYSTEM DESIGN DEEP DIVE

### 2.1 High-Level Architecture

**DoorDash Product Search Architecture**:
```mermaid
graph TD
    User[User Query] --> QueryProc[Query Understanding (LLM)]
    QueryProc --> Intent{Intent Classification}
    
    Intent -- "Dish/Item" --> Semantic[Semantic Retrieval (Vector)]
    Intent -- "Restaurant" --> Lexical[Lexical Retrieval (Keyword)]
    
    Semantic --> Candidates
    Lexical --> Candidates
    
    subgraph "Knowledge Graph"
        RawMenu[Raw Menu Data] --> LLM_Extract[LLM Extraction]
        LLM_Extract --> Entities[Standardized Entities]
        Entities --> KG[(Product Knowledge Graph)]
        KG --> Embed[Embedding Generation]
    end
    
    Embed --> VectorDB[(Vector DB)]
    VectorDB --> Semantic
    
    Candidates --> GeoFilter[Geo-Spatial Filter]
    GeoFilter --> Availability[Real-Time Availability Check]
    Availability --> Ranking[LTR Ranking Model]
    Ranking --> Results
```

**Instacart Hybrid Retrieval Architecture**:
```mermaid
graph TD
    Query[User Query] --> Parallel[Parallel Execution]
    
    subgraph "Retrieval Paths"
        Parallel --> Path1[Lexical (Elasticsearch)]
        Parallel --> Path2[Semantic (Dense Retrieval)]
        
        Path1 --> TopK_Lex[Top-K Keyword Matches]
        Path2 --> TopK_Sem[Top-K Semantic Matches]
    end
    
    TopK_Lex --> Fusion[Hybrid Fusion]
    TopK_Sem --> Fusion
    
    Fusion --> Availability[Store-Specific Inventory Check]
    Availability --> ReRank[Personalized Re-Ranking]
    ReRank --> Final[Final Results]
```

### Tech Stack Identified

| Component | Technology/Tool | Purpose | Company |
|-----------|----------------|---------|---------|
| **LLM** | GPT-4 / Fine-tuned Llama | Query Understanding, KG Construction | DoorDash |
| **Vector DB** | Faiss / Milvus | Semantic Search | Instacart, DoorDash |
| **Search Engine** | Apache Lucene (Custom) | Core Indexing | DoorDash (Moved from ES) |
| **Search Engine** | Elasticsearch | Lexical Search | Instacart |
| **Embedding Model** | MiniLM-L3-v2 | Product/Query Embeddings | Instacart |
| **Feature Store** | Redis / Cassandra | Real-time Availability | All |
| **Ranking Model** | XGBoost / LambdaMART | Learning to Rank | DoorDash, Uber Eats |

### 2.2 Data Pipeline

**DoorDash (Knowledge Graph Construction)**:
- **Ingestion**: Raw menu text from merchants (often messy, unstructured).
- **LLM Processing**:
    - **Extraction**: Identify "Burger", "Fries", "Coke" from "Combo #1".
    - **Standardization**: Map "Chkn Brgr" -> "Chicken Burger".
    - **Tagging**: Add tags like "Gluten-Free", "Spicy", "Vegan" based on description.
- **Graph Building**: Link items to categories, cuisines, and dietary restrictions.
- **Embedding**: Generate vectors for standardized entities.

**Instacart (Inventory Sync)**:
- **Source**: Retailer inventory feeds (EDI, API) + Real-time picker feedback ("Item not found").
- **Indexing**:
    - **Base Index**: Static product data (Name, Image, Brand).
    - **Dynamic Overlay**: Store-specific availability map.
- **Search Time**: Retrieve global products -> Filter by store availability mask.

### 2.3 Feature Engineering

**Key Features**:

**DoorDash**:
- **Geo-Spatial**: Distance to user, estimated delivery time (ETA).
- **Restaurant**: Rating, cuisine type, price range, "DashPass" eligibility.
- **Item**: Image quality score, popularity, match score with query.
- **Personalization**: User's past cuisine preferences, average order value.

**Instacart**:
- **Substitution**: "Substituteability Score" (How likely is B a good sub for A?).
- **Co-occurrence**: "Users who bought X also bought Y" (Word2Vec style).
- **Freshness**: Last time item was confirmed in stock.

### 2.4 Model Architecture

**Instacart's Hybrid Retrieval**:
- **Lexical**: BM25 algorithm. Good for exact brand matches ("Tropicana Orange Juice").
- **Semantic**: Bi-encoder (MiniLM). Good for concepts ("Healthy Snacks", "Keto Breakfast").
- **Fusion**: Linear combination of scores: `Score = w1 * BM25 + w2 * CosineSim`.
    - *Optimization*: Weights `w1` and `w2` are tuned based on query entropy (ambiguity).

**DoorDash's LLM-Enhanced Retrieval**:
- **Query Expansion**: LLM rewrites "Vegan protein" -> "Tofu", "Seitan", "Tempeh", "Lentils".
- **Entity Linking**: Maps query terms to Knowledge Graph nodes.
- **Ranking**: Learning-to-Rank (LTR) model trained on click logs + "Add to Cart" events.

---

## PART 3: MLOPS & INFRASTRUCTURE

### 3.1 Model Deployment & Serving

**Real-Time Constraints**:
- **Inventory Check**: Must happen *after* retrieval but *before* ranking.
    - *Why?* Ranking out-of-stock items high frustrates users.
    - *Challenge*: Checking availability for 1000 candidates adds latency.
    - *Solution*: **Bitsets**. Store availability as a bitset in memory (Redis) for fast filtering.

**Geo-Hashing**:
- **S2 Geometry / H3**: Index restaurants/stores by geo-cells.
- **Retrieval**: Query only the cells covering the user's delivery radius.

### 3.2 Monitoring & Observability

**Metrics**:
- **Null Results Rate**: Critical for delivery. "No restaurants found" is a churn driver.
- **Cart Add Rate**: The ultimate relevance metric.
- **Substitution Acceptance Rate**: Did the user accept the suggested replacement?

**Drift**:
- **Menu Drift**: Restaurants change menus seasonally.
- **Price Drift**: Prices change dynamically.

### 3.3 Operational Challenges

**The "Banana" Problem (Instacart)**:
- **Issue**: "Banana" is the most popular item. Everyone searches for it.
- **Challenge**: If the semantic model drifts, "Banana" might match "Plantain" or "Banana Bread".
- **Fix**: **Hard-coded boosts** or "Pinning" for top head queries to ensure exact matches.

**The "Closed Store" Problem (DoorDash)**:
- **Issue**: A restaurant is relevant but currently closed.
- **UI Decision**: Show it at the bottom (grayed out) or hide it?
- **ML Impact**: Ranking models need to know "Open/Closed" status as a strong feature.

---

## PART 4: EVALUATION & VALIDATION

### 4.1 Offline Evaluation

**Datasets**:
- **Click Logs**: (Query, Impression, Click, Convert).
- **Human Labels**: Raters judge "Is this item a good substitute?"

**Metrics**:
- **Recall@K**: Are the relevant restaurants in the top K?
- **NDCG**: Is the best restaurant at position 1?

### 4.2 Online Evaluation

**Switchback Testing**:
- **Method**: Switch algorithms every hour (or geo-region) to control for time-of-day effects (Lunch vs. Dinner).
- **Why?**: Standard A/B testing is noisy because lunch behavior != dinner behavior.

### 4.3 Failure Cases

- **"Ghost Kitchens"**: Multiple "restaurants" serving the same food from one kitchen. Clutters search results.
    - *Fix*: Entity resolution to group virtual brands.
- **Vague Queries**: "Dinner" -> Too broad.
    - *Fix*: Show diverse categories (Pizza, Sushi, Burger) instead of specific items.

---

## PART 5: LESSONS LEARNED & KEY TAKEAWAYS

### 5.1 Technical Insights

1.  **LLMs Clean the Data**: DoorDash proved that LLMs are best used *offline* to clean and structure messy menu data into a Knowledge Graph, rather than just for online query processing.
2.  **Hybrid is Essential**: Instacart showed that for grocery, you cannot drop Lexical search. Brand names are too specific for pure vector search.
3.  **Geo-Filtering First**: In delivery, location is the primary constraint. Filter by Geo -> Then Retrieve -> Then Rank.

### 5.2 Operational Insights

1.  **Availability is King**: The best ranking algorithm fails if the item is out of stock. Real-time inventory sync is the backbone of delivery search.
2.  **Substitution is a Search Problem**: Finding a substitute is essentially "Search for similar item excluding original". The same embeddings can be used for both.

---

## PART 6: REFERENCE ARCHITECTURE (DELIVERY SEARCH)

```mermaid
graph TD
    subgraph "Offline Pipeline"
        Menu[Menu Data] --> LLM[LLM Cleaner]
        LLM --> KG[Knowledge Graph]
        KG --> Embed[Embedding Model]
        Embed --> VectorDB[(Vector DB)]
    end

    subgraph "Online Pipeline"
        User --> QueryProc[Query Processor]
        QueryProc --> Geo[Geo-Filter (H3/S2)]
        Geo --> Candidates[Candidate Stores]
        
        Candidates --> Parallel{Parallel Retrieval}
        Parallel --> Lex[Lexical Search]
        Parallel --> Sem[Semantic Search]
        
        Lex --> TopK
        Sem --> TopK
        
        TopK --> Fusion[Hybrid Fusion]
        Fusion --> Inventory[Real-Time Inventory Check]
        Inventory --> Rank[LTR Ranking]
        Rank --> Results
    end
```

### Estimated Costs
- **LLM Costs**: High for initial KG build (processing millions of items). Low for maintenance.
- **Infrastructure**: High. Real-time inventory checks require massive Redis clusters.
- **Team**: 8-12 Engineers (Heavy on Backend/Systems due to real-time constraints).

---

*Analysis completed: November 2025*
