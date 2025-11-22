# Week 2: Transformer Architecture & Attention Mechanisms
## Summary

This week explores the Transformer architecture that revolutionized NLP and enabled modern LLMs.

### Key Concepts

**Day 8: Self-Attention Mechanism**
- Scaled dot-product attention
- Query, Key, Value matrices
- Attention weights computation
- Computational complexity O(n²)

**Day 9: Multi-Head Attention**
- Multiple attention heads in parallel
- Head concatenation and projection
- Why multiple heads help
- Positional information flow

**Day 10: Transformer Encoder Architecture**
- Layer structure: Attention + FFN
- Residual connections and layer normalization
- Pre-norm vs post-norm
- BERT architecture deep dive

**Day 11: Transformer Decoder Architecture**  
- Masked self-attention
- Cross-attention to encoder
- Autoregressive generation
- GPT architecture deep dive

**Day 12: Positional Encodings**
- Sinusoidal positional encodings
- Learned positional embeddings
- Relative positional encodings (T5)
- RoPE and ALiBi (modern approaches)

**Day 13: Transformer Training & Optimization**
- Learning rate schedules for Transformers
- Training stability techniques
- Gradient checkpointing for memory
- Pre-training objectives (MLM, CLM)

**Day 14: Transformer Variants & Innovations**
- Efficient attention mechanisms
- Longformer, Big Bird (sparse attention)
- Flash Attention
- Modern architectural improvements

### Labs

Practical exercises for Week 2:

1. **Implementing Attention from Scratch:**
   - Build scaled dot-product attention
   - Multi-head attention module
   - Visualize attention weights

2. **Building a Mini-Transformer:**
   - Encoder-only (BERT-style)
   - Decoder-only (GPT-style)
   - Full encoder-decoder (T5-style)

3. **Positional Encoding Experiments:**
   - Compare sinusoidal vs learned
   - Test extrapolation on longer sequences
   - Implement RoPE

4. **Training a Small Language Model:**
   - Train GPT-2 small on WikiText
   - Implement gradient checkpointing
   - Monitor attention patterns

5. **Attention Visualization:**
   - Extract and visualize attention weights
   - Analyze what different heads learn
   - Create attention heatmaps

6. **Performance Optimization:**
   - Profile Transformer forward/backward
   - Implement Flash Attention
   - Benchmark different implementations

### Key Takeaways

- **Attention is All You Need**: Self-attention replaces recurrence
- **Parallelization**: Process entire sequence simultaneously (vs sequential RNNs)
- **Scalability**: Architecture scales to billions of parameters
- **Flexibility**: Same architecture for encoding, decoding, seq2seq
- **Position information**: Must be explicitly added (no inherent order)
- **Computational cost**: O(n²) in sequence length (key limitation)

### From Theory to Practice

Week 2 bridges theoretical understanding with practical implementation:
- Mathematical foundations of attention
- Architectural design choices and trade-offs
- Modern optimizations (Flash Attention, efficient variants)
- Training large Transformers at scale

### What's Next

Week 3 dives into Large Language Model architecture and training:
- Scaling laws
- Pre-training strategies  
- Model architectures (GPT, BERT, T5)
- Training massive models efficiently

### Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [RoFormer: Enhanced with RoPE](https://arxiv.org/abs/2104.09864)
