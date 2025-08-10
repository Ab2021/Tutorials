# Day 4.3: Text Processing and NLP for Recommendations

## Learning Objectives
By the end of this session, you will:
- Master advanced NLP techniques for content-based recommendations
- Implement semantic similarity using word embeddings and transformers
- Apply topic modeling for content understanding and clustering
- Use sentiment analysis for review-based recommendation features
- Handle multi-language content processing for global recommendations

## 1. Advanced Text Preprocessing

### Comprehensive Text Preprocessing Pipeline

```python
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from collections import Counter
import numpy as np

class AdvancedTextPreprocessor:
    """
    Comprehensive text preprocessing for recommendation systems
    """
    
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Load spaCy model for advanced processing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: spaCy model not found. Some features will be limited.")
            self.nlp = None
        
        # Custom domain-specific stopwords
        self.domain_stopwords = set()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.html_pattern = re.compile(r'<.*?>')
        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')
        
    def preprocess_text(self, text, steps=None):
        """
        Apply comprehensive text preprocessing
        
        Args:
            text: Input text string
            steps: List of preprocessing steps to apply
                  Options: ['lowercase', 'remove_html', 'remove_urls', 
                           'remove_emails', 'remove_numbers', 'remove_punctuation',
                           'tokenize', 'remove_stopwords', 'lemmatize', 'stem']
        
        Returns:
            Processed text or list of tokens
        """
        if steps is None:
            steps = [
                'lowercase', 'remove_html', 'remove_urls', 'remove_emails',
                'remove_numbers', 'remove_punctuation', 'tokenize',
                'remove_stopwords', 'lemmatize'
            ]
        
        processed_text = text
        
        # Apply preprocessing steps in sequence
        for step in steps:
            if step == 'lowercase':
                processed_text = processed_text.lower()
            elif step == 'remove_html':
                processed_text = self._remove_html_tags(processed_text)
            elif step == 'remove_urls':
                processed_text = self._remove_urls(processed_text)
            elif step == 'remove_emails':
                processed_text = self._remove_emails(processed_text)
            elif step == 'remove_numbers':
                processed_text = self._remove_numbers(processed_text)
            elif step == 'remove_punctuation':
                processed_text = self._remove_punctuation(processed_text)
            elif step == 'tokenize':
                processed_text = self._tokenize(processed_text)
            elif step == 'remove_stopwords':
                if isinstance(processed_text, list):
                    processed_text = self._remove_stopwords(processed_text)
                else:
                    # Tokenize first if not already done
                    tokens = self._tokenize(processed_text)
                    processed_text = self._remove_stopwords(tokens)
            elif step == 'lemmatize':
                if isinstance(processed_text, list):
                    processed_text = self._lemmatize_tokens(processed_text)
                else:
                    tokens = self._tokenize(processed_text)
                    processed_text = self._lemmatize_tokens(tokens)
            elif step == 'stem':
                if isinstance(processed_text, list):
                    processed_text = self._stem_tokens(processed_text)
                else:
                    tokens = self._tokenize(processed_text)
                    processed_text = self._stem_tokens(tokens)
        
        return processed_text
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_pos_tags(self, text):
        """Extract part-of-speech tags"""
        if self.nlp is None:
            # Fallback to NLTK
            tokens = word_tokenize(text)
            return nltk.pos_tag(tokens)
        
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def extract_noun_phrases(self, text):
        """Extract noun phrases from text"""
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            noun_phrases.append({
                'text': chunk.text,
                'root': chunk.root.text,
                'start': chunk.start_char,
                'end': chunk.end_char
            })
        
        return noun_phrases
    
    def add_domain_stopwords(self, words):
        """Add domain-specific stopwords"""
        self.domain_stopwords.update(words)
    
    def _remove_html_tags(self, text):
        """Remove HTML tags"""
        return self.html_pattern.sub('', text)
    
    def _remove_urls(self, text):
        """Remove URLs"""
        return self.url_pattern.sub('', text)
    
    def _remove_emails(self, text):
        """Remove email addresses"""
        return self.email_pattern.sub('', text)
    
    def _remove_numbers(self, text):
        """Remove numbers"""
        return self.number_pattern.sub('', text)
    
    def _remove_punctuation(self, text):
        """Remove punctuation"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _tokenize(self, text):
        """Tokenize text"""
        return word_tokenize(text)
    
    def _remove_stopwords(self, tokens):
        """Remove stopwords"""
        all_stopwords = self.stop_words.union(self.domain_stopwords)
        return [token for token in tokens if token.lower() not in all_stopwords]
    
    def _lemmatize_tokens(self, tokens):
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def _stem_tokens(self, tokens):
        """Stem tokens"""
        return [self.stemmer.stem(token) for token in tokens]

class LanguageDetector:
    """Simple language detection for multi-language content"""
    
    def __init__(self):
        # Language-specific stopwords for detection
        self.language_stopwords = {
            'english': {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'as', 'are'},
            'spanish': {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se'},
            'french': {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'},
            'german': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'},
            'italian': {'di', 'che', 'e', 'il', 'un', 'a', 'è', 'per', 'una', 'in'}
        }
    
    def detect_language(self, text):
        """
        Simple language detection based on stopword frequency
        
        Args:
            text: Input text
            
        Returns:
            Detected language string
        """
        words = text.lower().split()
        word_set = set(words)
        
        scores = {}
        for lang, stopwords in self.language_stopwords.items():
            # Count overlap with language-specific stopwords
            overlap = len(word_set.intersection(stopwords))
            scores[lang] = overlap / len(stopwords)
        
        # Return language with highest score
        return max(scores, key=scores.get)
```

## 2. Word Embeddings and Semantic Similarity

### Word2Vec and FastText Implementation

```python
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

class EmbeddingBasedSimilarity:
    """
    Semantic similarity using word embeddings
    """
    
    def __init__(self, embedding_dim=100, min_count=2, window=5):
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.window = window
        
        self.word2vec_model = None
        self.fasttext_model = None
        self.doc2vec_model = None
        
        self.preprocessor = AdvancedTextPreprocessor()
    
    def train_word2vec(self, documents):
        """
        Train Word2Vec model on documents
        
        Args:
            documents: List of text documents
        """
        # Preprocess documents
        processed_docs = []
        for doc in documents:
            tokens = self.preprocessor.preprocess_text(
                doc, steps=['lowercase', 'tokenize', 'remove_stopwords']
            )
            processed_docs.append(tokens)
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=processed_docs,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=1  # Skip-gram
        )
        
        return self.word2vec_model
    
    def train_fasttext(self, documents):
        """
        Train FastText model on documents
        
        Args:
            documents: List of text documents
        """
        # Preprocess documents
        processed_docs = []
        for doc in documents:
            tokens = self.preprocessor.preprocess_text(
                doc, steps=['lowercase', 'tokenize', 'remove_stopwords']
            )
            processed_docs.append(tokens)
        
        # Train FastText model
        self.fasttext_model = FastText(
            sentences=processed_docs,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=1  # Skip-gram
        )
        
        return self.fasttext_model
    
    def train_doc2vec(self, documents):
        """
        Train Doc2Vec model for document-level embeddings
        
        Args:
            documents: List of text documents
        """
        # Prepare tagged documents
        tagged_docs = []
        for i, doc in enumerate(documents):
            tokens = self.preprocessor.preprocess_text(
                doc, steps=['lowercase', 'tokenize', 'remove_stopwords']
            )
            tagged_docs.append(TaggedDocument(words=tokens, tags=[str(i)]))
        
        # Train Doc2Vec model
        self.doc2vec_model = Doc2Vec(
            documents=tagged_docs,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            dm=1  # Distributed Memory
        )
        
        return self.doc2vec_model
    
    def get_word_similarity(self, word1, word2, model_type='word2vec'):
        """
        Compute similarity between two words
        
        Args:
            word1, word2: Words to compare
            model_type: 'word2vec' or 'fasttext'
            
        Returns:
            Similarity score (0-1)
        """
        try:
            if model_type == 'word2vec' and self.word2vec_model:
                return self.word2vec_model.wv.similarity(word1, word2)
            elif model_type == 'fasttext' and self.fasttext_model:
                return self.fasttext_model.wv.similarity(word1, word2)
            else:
                return 0.0
        except KeyError:
            return 0.0
    
    def get_document_vector(self, document, method='average'):
        """
        Get document-level vector representation
        
        Args:
            document: Text document
            method: 'average', 'tfidf_weighted', 'doc2vec'
            
        Returns:
            Document vector
        """
        if method == 'doc2vec' and self.doc2vec_model:
            tokens = self.preprocessor.preprocess_text(
                document, steps=['lowercase', 'tokenize', 'remove_stopwords']
            )
            return self.doc2vec_model.infer_vector(tokens)
        
        # For Word2Vec/FastText methods
        model = self.word2vec_model or self.fasttext_model
        if not model:
            return np.zeros(self.embedding_dim)
        
        tokens = self.preprocessor.preprocess_text(
            document, steps=['lowercase', 'tokenize', 'remove_stopwords']
        )
        
        if method == 'average':
            return self._average_word_vectors(tokens, model)
        elif method == 'tfidf_weighted':
            return self._tfidf_weighted_vectors(tokens, model, document)
        
        return np.zeros(self.embedding_dim)
    
    def compute_document_similarity(self, doc1, doc2, method='average'):
        """
        Compute similarity between two documents
        
        Args:
            doc1, doc2: Text documents
            method: Vector aggregation method
            
        Returns:
            Similarity score (0-1)
        """
        vec1 = self.get_document_vector(doc1, method)
        vec2 = self.get_document_vector(doc2, method)
        
        # Compute cosine similarity
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return similarity
    
    def find_similar_words(self, word, top_n=10, model_type='word2vec'):
        """
        Find most similar words to given word
        
        Args:
            word: Input word
            top_n: Number of similar words to return
            model_type: 'word2vec' or 'fasttext'
            
        Returns:
            List of (word, similarity) tuples
        """
        try:
            if model_type == 'word2vec' and self.word2vec_model:
                similar_words = self.word2vec_model.wv.most_similar(word, topn=top_n)
            elif model_type == 'fasttext' and self.fasttext_model:
                similar_words = self.fasttext_model.wv.most_similar(word, topn=top_n)
            else:
                return []
            
            return similar_words
        except KeyError:
            return []
    
    def _average_word_vectors(self, tokens, model):
        """Average word vectors to get document vector"""
        vectors = []
        for token in tokens:
            try:
                vector = model.wv[token]
                vectors.append(vector)
            except KeyError:
                continue
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.embedding_dim)
    
    def _tfidf_weighted_vectors(self, tokens, model, document):
        """TF-IDF weighted average of word vectors"""
        # Simple TF-IDF calculation
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        vectors = []
        weights = []
        
        for token in set(tokens):
            try:
                vector = model.wv[token]
                tf = token_counts[token] / total_tokens
                # Simplified IDF (would use proper corpus statistics in practice)
                idf = 1.0  # Placeholder
                weight = tf * idf
                
                vectors.append(vector * weight)
                weights.append(weight)
            except KeyError:
                continue
        
        if vectors:
            return np.sum(vectors, axis=0) / (sum(weights) + 1e-10)
        else:
            return np.zeros(self.embedding_dim)

# Transformer-based embeddings
class TransformerEmbeddings:
    """
    Modern transformer-based embeddings for semantic similarity
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError:
            print("Warning: sentence-transformers not installed")
            self.available = False
    
    def encode_documents(self, documents):
        """
        Encode documents to dense vectors
        
        Args:
            documents: List of text documents
            
        Returns:
            Document embeddings matrix
        """
        if not self.available:
            return np.array([])
        
        # Encode all documents at once for efficiency
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        return embeddings
    
    def compute_similarity_matrix(self, documents):
        """
        Compute pairwise similarity matrix for documents
        
        Args:
            documents: List of text documents
            
        Returns:
            Similarity matrix
        """
        embeddings = self.encode_documents(documents)
        if embeddings.size == 0:
            return np.array([])
        
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def find_most_similar(self, query_doc, candidate_docs, top_k=10):
        """
        Find most similar documents to query
        
        Args:
            query_doc: Query document text
            candidate_docs: List of candidate documents
            top_k: Number of similar documents to return
            
        Returns:
            List of (doc_index, similarity_score) tuples
        """
        if not self.available:
            return []
        
        # Encode query and candidates
        query_embedding = self.model.encode([query_doc])
        candidate_embeddings = self.model.encode(candidate_docs)
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices]
        
        return results
```

## 3. Topic Modeling for Content Understanding

### Latent Dirichlet Allocation (LDA) Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class TopicModelingEngine:
    """
    Topic modeling for content understanding and clustering
    """
    
    def __init__(self, n_topics=10, max_features=1000):
        self.n_topics = n_topics
        self.max_features = max_features
        
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100,
            learning_method='online'
        )
        
        self.preprocessor = AdvancedTextPreprocessor()
        self.fitted = False
        self.feature_names = []
        
    def fit_transform(self, documents):
        """
        Fit topic model and transform documents
        
        Args:
            documents: List of text documents
            
        Returns:
            Document-topic matrix
        """
        # Preprocess documents
        processed_docs = []
        for doc in documents:
            processed_text = self.preprocessor.preprocess_text(
                doc, steps=['lowercase', 'remove_html', 'remove_urls']
            )
            if isinstance(processed_text, list):
                processed_text = ' '.join(processed_text)
            processed_docs.append(processed_text)
        
        # Vectorize documents
        doc_term_matrix = self.vectorizer.fit_transform(processed_docs)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit and transform with LDA
        doc_topic_matrix = self.lda_model.fit_transform(doc_term_matrix)
        self.fitted = True
        
        return doc_topic_matrix
    
    def transform(self, documents):
        """Transform new documents to topic space"""
        if not self.fitted:
            raise ValueError("Model must be fitted before transform")
        
        # Preprocess documents
        processed_docs = []
        for doc in documents:
            processed_text = self.preprocessor.preprocess_text(
                doc, steps=['lowercase', 'remove_html', 'remove_urls']
            )
            if isinstance(processed_text, list):
                processed_text = ' '.join(processed_text)
            processed_docs.append(processed_text)
        
        # Vectorize and transform
        doc_term_matrix = self.vectorizer.transform(processed_docs)
        doc_topic_matrix = self.lda_model.transform(doc_term_matrix)
        
        return doc_topic_matrix
    
    def get_topic_words(self, topic_id, top_n=10):
        """
        Get top words for a specific topic
        
        Args:
            topic_id: Topic index
            top_n: Number of top words to return
            
        Returns:
            List of (word, weight) tuples
        """
        if not self.fitted:
            return []
        
        topic_words = []
        topic_weights = self.lda_model.components_[topic_id]
        top_indices = np.argsort(topic_weights)[::-1][:top_n]
        
        for idx in top_indices:
            word = self.feature_names[idx]
            weight = topic_weights[idx]
            topic_words.append((word, weight))
        
        return topic_words
    
    def get_document_topics(self, doc_topic_matrix, doc_idx, top_n=3):
        """
        Get top topics for a specific document
        
        Args:
            doc_topic_matrix: Document-topic matrix from fit_transform
            doc_idx: Document index
            top_n: Number of top topics to return
            
        Returns:
            List of (topic_id, probability) tuples
        """
        doc_topics = doc_topic_matrix[doc_idx]
        top_topic_indices = np.argsort(doc_topics)[::-1][:top_n]
        
        result = []
        for topic_id in top_topic_indices:
            probability = doc_topics[topic_id]
            result.append((topic_id, probability))
        
        return result
    
    def compute_topic_similarity(self, doc_topic_matrix1, doc_topic_matrix2):
        """
        Compute topic-based similarity between document sets
        
        Args:
            doc_topic_matrix1, doc_topic_matrix2: Document-topic matrices
            
        Returns:
            Similarity matrix
        """
        return cosine_similarity(doc_topic_matrix1, doc_topic_matrix2)
    
    def print_topics(self, top_n=10):
        """Print all topics with their top words"""
        if not self.fitted:
            print("Model not fitted yet")
            return
        
        for topic_id in range(self.n_topics):
            topic_words = self.get_topic_words(topic_id, top_n)
            words_str = ', '.join([word for word, _ in topic_words])
            print(f"Topic {topic_id}: {words_str}")
    
    def visualize_topics(self, save_path=None):
        """Create word clouds for topics"""
        if not self.fitted:
            return
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        for topic_id in range(min(10, self.n_topics)):
            topic_words = self.get_topic_words(topic_id, 50)
            word_freq = {word: weight for word, weight in topic_words}
            
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white'
            ).generate_from_frequencies(word_freq)
            
            axes[topic_id].imshow(wordcloud, interpolation='bilinear')
            axes[topic_id].set_title(f'Topic {topic_id}')
            axes[topic_id].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

class HierarchicalTopicModeling:
    """
    Hierarchical topic modeling for multi-level content organization
    """
    
    def __init__(self, levels=2, topics_per_level=None):
        self.levels = levels
        self.topics_per_level = topics_per_level or [10, 5]  # Topics at each level
        self.topic_models = {}  # Level -> TopicModelingEngine
        self.hierarchy = {}  # Parent topic -> child topics mapping
    
    def fit_hierarchical_topics(self, documents):
        """
        Fit hierarchical topic model
        
        Args:
            documents: List of text documents
            
        Returns:
            Hierarchical topic structure
        """
        current_docs = documents
        current_assignments = list(range(len(documents)))
        
        for level in range(self.levels):
            n_topics = self.topics_per_level[level]
            
            # Fit topic model for current level
            topic_model = TopicModelingEngine(n_topics=n_topics)
            doc_topic_matrix = topic_model.fit_transform(current_docs)
            
            self.topic_models[level] = topic_model
            
            # Assign documents to dominant topics
            dominant_topics = np.argmax(doc_topic_matrix, axis=1)
            
            if level < self.levels - 1:
                # Group documents by topic for next level
                topic_doc_groups = {}
                for doc_idx, topic_id in enumerate(dominant_topics):
                    if topic_id not in topic_doc_groups:
                        topic_doc_groups[topic_id] = []
                    topic_doc_groups[topic_id].append(current_docs[doc_idx])
                
                # Store hierarchy information
                for parent_topic, child_docs in topic_doc_groups.items():
                    if len(child_docs) > 1:  # Only process if enough documents
                        self.hierarchy[(level, parent_topic)] = child_docs
        
        return self.hierarchy
    
    def get_document_hierarchy(self, document):
        """Get hierarchical topic assignment for a document"""
        hierarchy_path = []
        current_doc = [document]
        
        for level in range(self.levels):
            if level in self.topic_models:
                topic_model = self.topic_models[level]
                doc_topic_matrix = topic_model.transform(current_doc)
                dominant_topic = np.argmax(doc_topic_matrix[0])
                hierarchy_path.append((level, dominant_topic))
        
        return hierarchy_path
```

## 4. Sentiment Analysis for Reviews

### Review Sentiment Analysis System

```python
import re
from textblob import TextBlob
from collections import defaultdict

class ReviewSentimentAnalyzer:
    """
    Sentiment analysis system for review-based recommendations
    """
    
    def __init__(self):
        self.aspect_keywords = {
            'quality': ['quality', 'build', 'material', 'construction', 'durable', 'sturdy'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth'],
            'service': ['service', 'support', 'staff', 'help', 'customer', 'delivery'],
            'usability': ['easy', 'difficult', 'user-friendly', 'intuitive', 'complex', 'simple'],
            'design': ['design', 'look', 'appearance', 'style', 'beautiful', 'ugly']
        }
        
        self.sentiment_intensifiers = {
            'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'incredibly': 2.0,
            'somewhat': 0.7, 'slightly': 0.5, 'kind of': 0.6, 'sort of': 0.6
        }
        
        self.negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere'}
    
    def analyze_review_sentiment(self, review_text):
        """
        Comprehensive sentiment analysis of review
        
        Args:
            review_text: Review text string
            
        Returns:
            Dictionary with sentiment scores and aspects
        """
        # Overall sentiment using TextBlob
        blob = TextBlob(review_text)
        overall_sentiment = blob.sentiment
        
        # Aspect-based sentiment analysis
        aspect_sentiments = self._analyze_aspect_sentiments(review_text)
        
        # Extract sentiment features
        sentiment_features = self._extract_sentiment_features(review_text)
        
        # Compute final sentiment score with adjustments
        adjusted_sentiment = self._adjust_sentiment_score(
            overall_sentiment.polarity, 
            review_text, 
            sentiment_features
        )
        
        return {
            'overall_polarity': overall_sentiment.polarity,
            'overall_subjectivity': overall_sentiment.subjectivity,
            'adjusted_sentiment': adjusted_sentiment,
            'aspect_sentiments': aspect_sentiments,
            'sentiment_features': sentiment_features,
            'sentiment_category': self._categorize_sentiment(adjusted_sentiment)
        }
    
    def extract_review_features(self, reviews):
        """
        Extract aggregated features from multiple reviews
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary of aggregated review features
        """
        if not reviews:
            return {}
        
        all_sentiments = []
        aspect_scores = defaultdict(list)
        feature_counts = defaultdict(int)
        
        for review in reviews:
            sentiment_analysis = self.analyze_review_sentiment(review)
            all_sentiments.append(sentiment_analysis['adjusted_sentiment'])
            
            # Collect aspect sentiments
            for aspect, score in sentiment_analysis['aspect_sentiments'].items():
                if score is not None:
                    aspect_scores[aspect].append(score)
            
            # Count sentiment features
            for feature, value in sentiment_analysis['sentiment_features'].items():
                feature_counts[feature] += value
        
        # Compute aggregated features
        features = {
            'avg_sentiment': np.mean(all_sentiments),
            'sentiment_std': np.std(all_sentiments),
            'positive_review_ratio': len([s for s in all_sentiments if s > 0.1]) / len(all_sentiments),
            'negative_review_ratio': len([s for s in all_sentiments if s < -0.1]) / len(all_sentiments),
            'review_count': len(reviews)
        }
        
        # Add aspect-specific features
        for aspect, scores in aspect_scores.items():
            if scores:
                features[f'{aspect}_sentiment_avg'] = np.mean(scores)
                features[f'{aspect}_sentiment_std'] = np.std(scores)
        
        # Add feature ratios
        total_reviews = len(reviews)
        for feature, count in feature_counts.items():
            features[f'{feature}_ratio'] = count / total_reviews
        
        return features
    
    def _analyze_aspect_sentiments(self, review_text):
        """Analyze sentiment for different aspects"""
        aspect_sentiments = {}
        sentences = review_text.split('.')
        
        for aspect, keywords in self.aspect_keywords.items():
            aspect_sentences = []
            
            # Find sentences mentioning this aspect
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    aspect_sentences.append(sentence)
            
            # Compute average sentiment for aspect
            if aspect_sentences:
                sentiments = []
                for sentence in aspect_sentences:
                    blob = TextBlob(sentence)
                    sentiments.append(blob.sentiment.polarity)
                aspect_sentiments[aspect] = np.mean(sentiments)
            else:
                aspect_sentiments[aspect] = None
        
        return aspect_sentiments
    
    def _extract_sentiment_features(self, review_text):
        """Extract sentiment-related features"""
        text_lower = review_text.lower()
        
        features = {
            'exclamation_count': review_text.count('!'),
            'question_count': review_text.count('?'),
            'caps_ratio': sum(1 for c in review_text if c.isupper()) / len(review_text),
            'has_negation': 1 if any(word in text_lower for word in self.negation_words) else 0,
            'has_intensifier': 1 if any(word in text_lower for word in self.sentiment_intensifiers) else 0,
            'review_length': len(review_text.split()),
            'sentence_count': len([s for s in review_text.split('.') if s.strip()])
        }
        
        # Positive/negative word counts (simplified)
        positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'poor'}
        
        words = text_lower.split()
        features['positive_word_count'] = sum(1 for word in words if word in positive_words)
        features['negative_word_count'] = sum(1 for word in words if word in negative_words)
        
        return features
    
    def _adjust_sentiment_score(self, base_sentiment, review_text, features):
        """Adjust sentiment score based on various factors"""
        adjusted = base_sentiment
        
        # Adjust for intensifiers
        if features['has_intensifier']:
            adjusted *= 1.2
        
        # Adjust for negations
        if features['has_negation']:
            adjusted *= 0.8
        
        # Adjust for exclamations (usually intensify sentiment)
        if features['exclamation_count'] > 0:
            adjustment = min(0.2, features['exclamation_count'] * 0.1)
            adjusted += adjustment if adjusted > 0 else -adjustment
        
        # Adjust for caps (usually indicates strong sentiment)
        if features['caps_ratio'] > 0.1:
            adjustment = min(0.15, features['caps_ratio'])
            adjusted += adjustment if adjusted > 0 else -adjustment
        
        # Ensure bounds
        adjusted = max(-1.0, min(1.0, adjusted))
        
        return adjusted
    
    def _categorize_sentiment(self, sentiment_score):
        """Categorize sentiment score into discrete categories"""
        if sentiment_score > 0.5:
            return 'very_positive'
        elif sentiment_score > 0.1:
            return 'positive'
        elif sentiment_score > -0.1:
            return 'neutral'
        elif sentiment_score > -0.5:
            return 'negative'
        else:
            return 'very_negative'

class ReviewQualityAssessor:
    """Assess review quality and helpfulness"""
    
    def __init__(self):
        self.quality_indicators = {
            'specific_details': ['specifically', 'detail', 'exactly', 'particular'],
            'comparisons': ['compare', 'better', 'worse', 'similar', 'unlike', 'versus'],
            'usage_context': ['used', 'using', 'after', 'during', 'while', 'when'],
            'recommendations': ['recommend', 'suggest', 'advise', 'should', 'would']
        }
    
    def assess_review_quality(self, review_text):
        """
        Assess the quality and helpfulness of a review
        
        Args:
            review_text: Review text string
            
        Returns:
            Quality score and features
        """
        features = self._extract_quality_features(review_text)
        quality_score = self._compute_quality_score(features)
        
        return {
            'quality_score': quality_score,
            'quality_features': features,
            'quality_category': self._categorize_quality(quality_score)
        }
    
    def _extract_quality_features(self, review_text):
        """Extract features indicating review quality"""
        text_lower = review_text.lower()
        words = text_lower.split()
        
        features = {
            'length': len(words),
            'sentence_count': len([s for s in review_text.split('.') if s.strip()]),
            'avg_sentence_length': len(words) / max(1, len([s for s in review_text.split('.') if s.strip()])),
            'has_specific_details': 0,
            'has_comparisons': 0,
            'has_usage_context': 0,
            'has_recommendations': 0,
            'spelling_errors': 0,  # Simplified
            'readability_score': 0  # Simplified
        }
        
        # Check for quality indicators
        for indicator_type, keywords in self.quality_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                features[f'has_{indicator_type}'] = 1
        
        # Simple spelling error detection (count words not in common dictionary)
        # This is simplified - would use proper spell checker in practice
        common_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        potential_errors = [word for word in words if len(word) > 3 and word not in common_words and not word.isdigit()]
        features['spelling_errors'] = min(5, len([w for w in potential_errors if len(w) < 15]))  # Cap at 5
        
        return features
    
    def _compute_quality_score(self, features):
        """Compute overall quality score"""
        score = 0.0
        
        # Length factor (optimal range: 20-200 words)
        length = features['length']
        if 20 <= length <= 200:
            score += 0.3
        elif 10 <= length < 20 or 200 < length <= 300:
            score += 0.15
        
        # Sentence structure
        if 2 <= features['sentence_count'] <= 10:
            score += 0.2
        
        # Quality indicators
        score += features['has_specific_details'] * 0.15
        score += features['has_comparisons'] * 0.1
        score += features['has_usage_context'] * 0.1
        score += features['has_recommendations'] * 0.1
        
        # Penalties
        score -= features['spelling_errors'] * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _categorize_quality(self, quality_score):
        """Categorize quality score"""
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.6:
            return 'good'
        elif quality_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
```

## 5. Multi-Language Content Processing

### Multi-Language Support System

```python
class MultiLanguageProcessor:
    """
    Handle multi-language content for global recommendations
    """
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.language_processors = {}
        self.translation_cache = {}
        
        # Language-specific preprocessing settings
        self.language_settings = {
            'english': {
                'stemmer': 'porter',
                'stopwords': 'english',
                'tokenizer': 'word_tokenize'
            },
            'spanish': {
                'stemmer': 'spanish',
                'stopwords': 'spanish',
                'tokenizer': 'word_tokenize'
            },
            'french': {
                'stemmer': 'french',
                'stopwords': 'french',
                'tokenizer': 'word_tokenize'
            }
        }
    
    def process_multilingual_content(self, content_items):
        """
        Process content items in multiple languages
        
        Args:
            content_items: List of {'text': str, 'language': str (optional)}
            
        Returns:
            Processed content with language metadata
        """
        processed_items = []
        
        for item in content_items:
            # Detect language if not provided
            if 'language' not in item:
                detected_lang = self.language_detector.detect_language(item['text'])
                item['language'] = detected_lang
            
            # Process text based on language
            processor = self._get_language_processor(item['language'])
            processed_text = processor.preprocess_text(item['text'])
            
            processed_items.append({
                'original_text': item['text'],
                'processed_text': processed_text,
                'language': item['language'],
                'features': self._extract_language_features(item['text'], item['language'])
            })
        
        return processed_items
    
    def translate_content(self, text, source_lang, target_lang='english'):
        """
        Simple translation (placeholder - would use proper translation service)
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Translated text
        """
        # Cache key
        cache_key = f"{source_lang}_{target_lang}_{hash(text)}"
        
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Placeholder translation logic
        # In practice, would use Google Translate API, Azure Translator, etc.
        if source_lang == target_lang:
            translated = text
        else:
            # Simple mock translation
            translated = f"[TRANSLATED FROM {source_lang.upper()}] {text}"
        
        self.translation_cache[cache_key] = translated
        return translated
    
    def create_multilingual_features(self, content_items, normalize_languages=True):
        """
        Create unified feature representation for multilingual content
        
        Args:
            content_items: Processed content items
            normalize_languages: Whether to translate to common language
            
        Returns:
            Unified feature matrix
        """
        if normalize_languages:
            # Translate all content to English for unified processing
            normalized_items = []
            for item in content_items:
                if item['language'] != 'english':
                    translated_text = self.translate_content(
                        item['original_text'], 
                        item['language'], 
                        'english'
                    )
                    normalized_items.append(translated_text)
                else:
                    normalized_items.append(item['original_text'])
            
            # Process normalized content
            processor = self._get_language_processor('english')
            feature_extractor = AdvancedTextFeatureExtractor()
            features = feature_extractor.fit_transform(normalized_items)
            
        else:
            # Language-specific processing
            language_groups = defaultdict(list)
            for i, item in enumerate(content_items):
                language_groups[item['language']].append((i, item['processed_text']))
            
            # Process each language group separately
            all_features = []
            for language, items in language_groups.items():
                indices, texts = zip(*items)
                
                feature_extractor = AdvancedTextFeatureExtractor()
                lang_features = feature_extractor.fit_transform(texts)
                
                # Store features with original indices
                for idx, features in zip(indices, lang_features):
                    all_features.append((idx, features))
            
            # Sort by original index and extract features
            all_features.sort(key=lambda x: x[0])
            features = np.array([f[1] for f in all_features])
        
        return features
    
    def _get_language_processor(self, language):
        """Get language-specific text processor"""
        if language not in self.language_processors:
            settings = self.language_settings.get(language, self.language_settings['english'])
            processor = AdvancedTextPreprocessor(language=settings['stopwords'])
            self.language_processors[language] = processor
        
        return self.language_processors[language]
    
    def _extract_language_features(self, text, language):
        """Extract language-specific features"""
        features = {
            'language': language,
            'text_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text)
        }
        
        # Language-specific features
        if language == 'english':
            features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        elif language == 'german':
            # German tends to have longer compound words
            features['compound_word_indicators'] = text.count('_') + text.count('-')
        elif language in ['spanish', 'french', 'italian']:
            # Romance languages feature detection
            features['accent_chars'] = len([c for c in text if ord(c) > 127])
        
        return features
```

## 6. Study Questions

### Beginner Level

1. What are the main steps in text preprocessing for recommendation systems?
2. How do word embeddings capture semantic similarity between words?
3. What is the difference between Word2Vec and FastText?
4. Explain how topic modeling can be used in content-based recommendations.
5. What are the key components of sentiment analysis for reviews?

### Intermediate Level

6. Implement a text similarity system using different embedding techniques and compare their performance.
7. How would you handle negations and intensifiers in sentiment analysis?
8. Design a topic modeling system that can handle short texts (like product titles).
9. What are the challenges in processing multi-language content for recommendations?
10. How would you extract aspect-based sentiments from product reviews?

### Advanced Level

11. Implement a hierarchical topic modeling system for multi-level content organization.
12. Design a system that can automatically detect and handle sarcasm in reviews.
13. How would you create domain-adaptive embeddings for specific recommendation domains?
14. Implement a cross-lingual content similarity system without translation.
15. Design a real-time text processing pipeline that can handle high-volume content updates.

### Tricky Questions

16. How would you handle concept drift in topic models for dynamic content collections?
17. Design a system that can identify fake or spam reviews using NLP techniques.
18. How would you create personalized topic models that adapt to individual user interests?
19. Implement a system that can extract structured information from unstructured review text.
20. How would you design a content similarity system that works across different content types (text, images, videos)?

## Key Takeaways

1. **Advanced preprocessing** is crucial for effective text-based recommendations
2. **Semantic embeddings** capture meaning beyond keyword matching
3. **Topic modeling** helps organize and understand large content collections
4. **Sentiment analysis** adds valuable signals from user-generated content
5. **Multi-language support** is essential for global recommendation systems
6. **Aspect-based analysis** provides fine-grained content understanding
7. **Quality assessment** helps weight different content sources appropriately

## Next Session Preview

In Day 4.4, we'll explore **Content Similarity and Matching Algorithms**, covering:
- Advanced similarity metrics beyond cosine similarity
- Locality-sensitive hashing for scalable similarity search
- Graph-based similarity measures
- Multi-modal content matching
- Real-time similarity computation techniques