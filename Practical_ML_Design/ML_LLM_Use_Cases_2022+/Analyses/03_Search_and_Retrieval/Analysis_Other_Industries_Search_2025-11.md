# ML Use Case Analysis: Search & Retrieval in Manufacturing, Media & Finance

**Analysis Date**: November 2025  
**Category**: Search & Retrieval  
**Industry**: Manufacturing, Media & Streaming, Finance  
**Articles Analyzed**: 5 (Netflix, Spotify, Monzo, Haleon)

---

## PART 1: USE CASE OVERVIEW

### 1.1 Basic Information

**Category**: Search & Retrieval  
**Industries**: Manufacturing (Consumer Health), Media & Streaming, Finance  
**Companies**: Netflix, Spotify, Monzo, Haleon  
**Years**: 2022-2025  
**Tags**: In-Video Search, Audio Search, Topic Modeling, Intent Understanding, Consumer Health

**Use Cases Analyzed**:
1. [Netflix - Building In-Video Search](https://netflixtechblog.com/building-in-video-search-936766f0017c) (2023)
2. [Spotify - Natural Language Search for Podcast Episodes](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/) (2022)
3. [Monzo - Using Topic Modelling to Understand Customer Saving Goals](https://monzo.com/blog/2023/08/01/using-topic-modelling-to-understand-customer-saving-goals) (2023)
4. [Haleon - Deriving Insights from Customer Queries](https://medium.com/haleon-engineering/deriving-insights-from-customer-queries-on-haleon-brands) (2023)

### 1.2 Problem Statement

**What business problem are they solving?**

These industries face **"Specialized Content"** search problems where the "document" is not text, or the "query" is highly ambiguous.

-   **Netflix (Media)**: Users remember a *moment* ("that scene where they kiss in the rain") but not the movie title. Standard metadata search fails here.
-   **Spotify (Media)**: Podcasts are 60-minute audio files. Users search for "productivity tips" buried in minute 42. Metadata (title/description) is insufficient.
-   **Monzo (Finance)**: Users create "Pots" (savings accounts) with names like "Rainy Day" or "The Big One". Monzo needs to map these vague names to specific financial goals (Emergency Fund, House Deposit) to offer relevant advice.
-   **Haleon (Manufacturing)**: Customers search for symptoms ("throbbing headache") or vague needs ("immune support"). Haleon must map these to specific products (Advil, Centrum) while navigating strict medical compliance.

**What makes this problem ML-worthy?**

1.  **Granularity**: Searching *inside* a video or audio file requires frame-level or segment-level indexing, increasing data volume by 1000x.
2.  **Multimodality**: Netflix must understand visual cues (rain, kiss) and audio cues (dialogue, music).
3.  **Ambiguity**: "Rainy Day" in Finance means "Emergency Fund", not "Weather". Context is everything.
4.  **Compliance**: In Health (Haleon) and Finance (Monzo), incorrect search results (e.g., suggesting wrong medication) have legal/safety consequences.

---

## PART 2: SYSTEM DESIGN DEEP DIVE

### 2.1 High-Level Architecture

**Netflix In-Video Search Architecture**:
```mermaid
graph TD
    Video[Video File] --> Split[Scene Splitter]
    
    subgraph "Multimodal Encoding"
        Split --> Visual[Visual Encoder (CLIP)]
        Split --> Audio[Audio Encoder (VGGish)]
        Split --> Text[Subtitle Encoder (BERT)]
    end
    
    Visual --> V_Emb
    Audio --> A_Emb
    Text --> T_Emb
    
    V_Emb & A_Emb & T_Emb --> Fusion[Late Fusion]
    Fusion --> SceneEmb[Scene Embedding]
    SceneEmb --> VectorDB[(Vector DB)]
    
    Query[User Query] --> Q_Encoder[Query Encoder]
    Q_Encoder --> Q_Emb
    Q_Emb --> ANN[ANN Search]
    VectorDB --> ANN
    ANN --> Results[Timestamped Scenes]
```

**Spotify Podcast Search Architecture**:
```mermaid
graph TD
    Podcast[Audio File] --> ASR[Speech-to-Text (Whisper)]
    ASR --> Transcript[Text Transcript]
    Transcript --> Segment[Segmentation]
    Segment --> Encoder[BERT Encoder]
    Encoder --> Vectors[Segment Vectors]
    Vectors --> Index[(Elasticsearch + KNN)]
    
    Query --> Q_Enc[Query Encoder]
    Q_Enc --> Search
    Index --> Search
    Search --> TopSegments
    TopSegments --> Agg[Episode Aggregation]
    Agg --> Results[Episodes]
```

### Tech Stack Identified

| Component | Technology/Tool | Purpose | Company |
|-----------|----------------|---------|---------|
| **ASR** | OpenAI Whisper / Google STT | Speech-to-Text | Spotify, Netflix |
| **Visual Model** | CLIP / ResNet | Frame Analysis | Netflix |
| **Topic Model** | BERTopic / LDA | Intent Clustering | Monzo, Haleon |
| **Vector DB** | Pinecone / Faiss | Embedding Storage | Netflix, Spotify |
| **NLP Model** | BioBERT / ClinicalBERT | Medical Entity Extraction | Haleon |
| **Orchestrator** | Airflow / Metaflow | Pipeline Management | All |

### 2.2 Data Pipeline

**Netflix (Scene Indexing)**:
-   **Shot Detection**: Algorithms detect cut boundaries to split video into "shots".
-   **Keyframe Extraction**: Select the most representative frame for each shot.
-   **Feature Extraction**: Run visual, audio, and text models on each shot.
-   **Indexing**: Store embeddings with metadata (Movie ID, Start Time, End Time).

**Monzo (Goal Classification)**:
-   **Input**: User-created Pot names ("Holiday 2024", "New Car").
-   **Preprocessing**: Remove emojis, handle slang.
-   **Embedding**: Convert names to vectors using a finance-tuned Sentence-BERT.
-   **Clustering**: Use HDBSCAN to find clusters of similar goals.
-   **Labeling**: Manually label clusters (e.g., Cluster 5 = "Wedding").

### 2.3 Feature Engineering

**Key Features**:

**Media (Netflix/Spotify)**:
-   **Temporal**: Time in video (Intro vs. Climax), Duration.
-   **Content**: Visual objects (Car, Gun), Audio events (Scream, Laughter), Spoken words.
-   **Meta**: Genre, Cast, Director.

**Finance/Health (Monzo/Haleon)**:
-   **Semantic**: Embedding of the query text.
-   **Context**: User age, location, transaction history (Monzo).
-   **Medical**: Symptom severity, duration (Haleon).

### 2.4 Model Architecture

**Haleon's Symptom Matcher**:
-   **NER (Named Entity Recognition)**: Extracts symptoms ("headache"), body parts ("head"), and modifiers ("throbbing").
-   **Relation Extraction**: Links symptoms to potential causes.
-   **Matching**: Maps extracted entities to a product ontology (Knowledge Graph).

**Spotify's "Two-Tower" for Audio**:
-   **Tower A (Query)**: Encodes user text.
-   **Tower B (Audio)**: Encodes spoken word transcripts + audio features (tempo, mood).
-   **Training**: Trained on (Search Query, Clicked Episode) pairs.

---

## PART 3: MLOPS & INFRASTRUCTURE

### 3.1 Model Deployment & Serving

**Offline vs. Online**:
-   **Media**: **Heavy Offline Processing**. Indexing a movie takes hours of GPU time. This is done once per title. Search is online and fast.
-   **Finance**: **Batch Processing**. Monzo classifies Pots nightly to update user insights. Real-time classification happens only during Pot creation.

**Latency**:
-   **Spotify**: <100ms. Searching millions of transcript segments requires highly optimized indices (inverted index for keywords + HNSW for vectors).

### 3.2 Monitoring & Observability

**Metrics**:
-   **Media**: "Play Rate" (Did user play the result?) and "Completion Rate" (Did they watch/listen to the end?).
-   **Finance**: "Conversion Rate" (Did user accept the savings advice?).
-   **Health**: "Safety Score" (Did we avoid recommending contraindicated products?).

### 3.3 Operational Challenges

**The "Context Window" Problem (Spotify)**:
-   **Issue**: A 1-minute segment might lack context. "He killed it!" could mean a murder mystery or a comedy set.
-   **Solution**: **Contextual Windowing**. Include previous/next 30 seconds in the embedding to capture context.

**The "Privacy" Problem (Monzo/Haleon)**:
-   **Issue**: Financial goals and health queries are highly sensitive.
-   **Solution**: **Differential Privacy** and **Anonymization**. Train models on aggregated data; never expose raw user queries to human labelers.

---

## PART 4: EVALUATION & VALIDATION

### 4.1 Offline Evaluation

**Golden Sets**:
-   **Netflix**: Editors manually tag scenes ("Kissing scenes", "Car chases"). Evaluate model recall against these tags.
-   **Haleon**: Medical professionals review query-product pairs for safety and accuracy.

### 4.2 Online Evaluation

**Implicit Feedback**:
-   **Spotify**: If a user searches, clicks a timestamp, and listens for >30s, it's a positive signal. If they seek immediately, it's a negative signal.

### 4.3 Failure Cases

-   **Metaphor vs. Literal (Netflix)**: User searches "Dark movie". Model returns movies with low lighting (literal) instead of "Dark Knight" or "Grim genre" (metaphor).
    -   *Fix*: Multitask learning to understand genre vs. visual attributes.

---

## PART 5: LESSONS LEARNED & KEY TAKEAWAYS

### 5.1 Technical Insights

1.  **Granularity Matters**: In media, indexing the *whole* file is useless. You must index *segments* (scenes/chapters).
2.  **ASR is the Unlock**: High-quality Speech-to-Text (Whisper) turned audio from a "black box" into searchable text, enabling standard NLP techniques on audio/video.
3.  **Domain Specificity**: Generic BERT fails on "medical" (Haleon) or "slang" (Monzo). Domain-tuned models (BioBERT, FinBERT) are essential.

### 5.2 Operational Insights

1.  **Safety First**: In Health/Finance, "No Result" is better than a "Wrong Result". Thresholds for retrieval must be high.
2.  **Human-in-the-Loop**: For specialized domains, human experts (doctors, editors) are needed to validate the "Ground Truth" for evaluation.

---

## PART 6: REFERENCE ARCHITECTURE (SPECIALIZED SEARCH)

```mermaid
graph TD
    subgraph "Ingestion (Offline)"
        Raw[Raw File (Video/Audio/Text)] --> Pre[Preprocessor]
        Pre -- "Media" --> Segment[Segmenter]
        Pre -- "Text" --> Clean[Cleaner]
        
        Segment --> ASR[ASR / Visual Model]
        ASR --> Features[Feature Vectors]
        Clean --> NLP[Domain NLP Model]
        NLP --> Features
        
        Features --> Index[(Vector Index)]
    end

    subgraph "Search (Online)"
        Query --> Encoder[Query Encoder]
        Encoder --> ANN[ANN Search]
        Index --> ANN
        ANN --> Filter[Safety/Compliance Filter]
        Filter --> Rank[Re-Ranker]
        Rank --> Results
    end
```

### Estimated Costs
-   **Processing**: Very High for Media (GPU hours for video decoding/encoding).
-   **Storage**: High. Vector indices for segmented media are huge (1000x larger than document indices).
-   **Team**: Specialized. Needs Computer Vision / Audio engineers (Media) or Domain Experts (Health/Finance).

---

*Analysis completed: November 2025*
