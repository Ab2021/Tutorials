# Week 1: AI/ML Fundamentals & Modern Tooling
## Summary

This week covers the foundational tools and concepts essential for modern AI engineering in 2025.

### Key Concepts

**Day 1: Modern AI Engineering Landscape**
- HuggingFace ecosystem (Transformers, Datasets, Accelerate, PEFT)
- PyTorch 2.0+ features (torch.compile, FSDP)
- Development environment best practices (Poetry, Docker)
- Production mindset for AI systems

**Day 2: Deep Learning Fundamentals for NLP**
- Neural network architectures for sequences
- Backpropagation and optimization (AdamW)
- Loss functions (cross-entropy for language modeling)
- Regularization (dropout, layer norm, weight decay)

**Day 3: Tokenization & Text Processing**
- BPE, WordPiece, SentencePiece algorithms
- Vocabulary construction and management
- Subword tokenization benefits
- Multilingual and code tokenization challenges

**Day 4: Embeddings & Representation Learning**
- Word embeddings (Word2Vec, GloVe, FastText)
- Contextualized embeddings (ELMo, BERT)
- Embedding dimensions and trade-offs
- Modern embedding techniques for LLMs

**Day 5: Neural Architectures for NLP**
- RNNs, LSTMs, GRUs for sequential processing
- CNNs for text classification
- Limitations that led to Transformers
- Historical context and evolution

**Day 6: Optimization & Training Techniques**
- Learning rate schedules (warmup, cosine decay)
- Gradient accumulation and mixed precision
- Training stability and convergence
- Monitoring and debugging training

**Day 7: GPUs, Mixed Precision, Distributed Training**
- GPU architecture and CUDA programming model
- FP16,  BF16, automatic mixed precision
- Data parallelism and basic model parallelism
- Multi-GPU training fundamentals

### Labs

The labs folder contains 15+ hands-on exercises covering:

1. **Environment Setup:**
   - Setting up Poetry/uv for ML projects
   - Creating Docker containers for reproducibility
   - HuggingFace Hub integration

2. **Training Basics:**
   - Implementing training loops in PyTorch
   - AdamW optimizer from scratch
   - Learning rate schedulers

3. **Tokenization:**
   - Training BPE tokenizer
   - Comparing tokenization strategies
   - Handling special tokens

4. **Embeddings:**
   - Training Word2Vec embeddings
   - Visualizing embeddings with t-SNE
   - Using pre-trained embeddings

5. **Optimization:**
   - Implementing gradient accumulation
   - Mixed precision training
   - Profiling and optimization

6. **Multi-GPU:**
   - Data Parallel training
   - Basic FSDP setup
   - Performance benchmarking

### Key Takeaways

- Modern AI engineering requires strong software engineering practices
- PyTorch 2.0 and HuggingFace provide production-ready infrastructure
- Understanding fundamentals (optimization, regularization) is critical for debugging
- Reproducibility (Docker, version control) is non-negotiable in production
- Mixed precision and distributed training enable scaling to large models

### Resources

- [HuggingFace Documentation](https://huggingface.co/docs)
- [PyTorch 2.0 Tutorials](https://pytorch.org/tutorials/)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### What's Next

Week 2 dives deep into the Transformer architecture and attention mechanisms - the foundation of all modern LLMs.
