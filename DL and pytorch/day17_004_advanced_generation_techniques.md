# Day 17.4: Advanced Generation Techniques - Sophisticated Text Generation and Control

## Overview

Advanced generation techniques represent the culmination of research into controllable, high-quality, and diverse text generation using large language models, encompassing sophisticated sampling strategies, guided generation methods, controllable text generation frameworks, and evaluation methodologies that enable precise control over model outputs while maintaining coherence and fluency. These techniques transform autoregressive language models from simple next-token predictors into versatile tools capable of producing text with specific stylistic properties, factual accuracy constraints, logical consistency requirements, and creative objectives tailored to diverse applications. The mathematical foundations underlying advanced sampling algorithms, the theoretical principles governing controllable generation, the optimization techniques for multi-objective text generation, and the practical methodologies for deploying sophisticated generation systems provide crucial insights into how large language models can be harnessed for complex real-world applications. Understanding these advanced techniques is essential for developing production-ready language generation systems, implementing sophisticated AI assistants, and pushing the boundaries of what automated text generation can achieve across domains ranging from creative writing to technical documentation.

## Sophisticated Sampling Strategies

### Beyond Standard Sampling Methods

**Limitations of Basic Sampling**
Traditional sampling methods have significant drawbacks:
- **Greedy decoding**: Deterministic but often repetitive and boring
- **Random sampling**: Diverse but potentially incoherent
- **Temperature scaling**: Global control but lacks nuanced adjustment
- **Top-k sampling**: Fixed cutoff regardless of probability distribution shape

**Mathematical Framework for Advanced Sampling**
General sampling objective:
$$\hat{y} = \arg\max_y \mathbb{E}_{x \sim P(x|y)}[\text{Quality}(x, y)] \cdot P(y|\text{context})$$

This balances model probability with desired quality metrics.

### Nucleus (Top-p) Sampling Deep Dive

**Dynamic Threshold Selection**
Top-p sampling adapts the vocabulary size based on probability mass:
$$V_p = \{w : \sum_{i=1}^{|V_p|} p_i \leq p\}$$

where tokens are sorted by probability: $p_1 \geq p_2 \geq ... \geq p_{|V|}$.

**Implementation Algorithm**:
```python
def nucleus_sampling(logits, p=0.9, temperature=1.0):
    # Apply temperature
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    
    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find nucleus
    nucleus_mask = cumulative_probs <= p
    
    # Ensure at least one token is selected
    nucleus_mask[0] = True
    
    # Filter probabilities
    filtered_probs = torch.zeros_like(probs)
    filtered_probs[sorted_indices[nucleus_mask]] = sorted_probs[nucleus_mask]
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    # Sample
    return torch.multinomial(filtered_probs, 1)
```

**Adaptive Nucleus Size**
The nucleus size automatically adjusts based on probability distribution:
- **Peaked distributions**: Small nucleus (high confidence)
- **Flat distributions**: Large nucleus (high uncertainty)

**Mathematical Analysis**:
$$|V_p| = \min\{k : \sum_{i=1}^{k} p_i \geq p\}$$

**Expected nucleus size**:
$$\mathbb{E}[|V_p|] = \sum_{k=1}^{|V|} P(|V_p| = k)$$

### Contrastive Search

**Motivation**
Balance model confidence with diversity to avoid repetitive text:
$$\text{Score}(x_t) = (1-\alpha) \cdot \log P(x_t|\mathbf{x}_{<t}) + \alpha \cdot \max_{i \in \mathbf{x}_{<t}} \cos(\mathbf{h}_{x_t}, \mathbf{h}_{x_i})$$

where:
- First term: Model probability (fluency)
- Second term: Maximum similarity to previous tokens (diversity penalty)
- $\alpha$: Balance parameter

**Algorithm Implementation**:
```python
def contrastive_search(model, input_ids, max_length, alpha=0.6, k=4):
    generated = input_ids.clone()
    past_hidden_states = []
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            hidden_states = outputs.hidden_states[-1][:, -1, :] # Last layer, last token
            
            # Get top-k candidates
            top_k_probs, top_k_indices = torch.topk(
                torch.softmax(logits, dim=-1), k
            )
            
            # Calculate contrastive scores
            scores = []
            for i, (prob, idx) in enumerate(zip(top_k_probs[0], top_k_indices[0])):
                model_score = torch.log(prob)
                
                if len(past_hidden_states) > 0:
                    # Get hidden state for this candidate
                    candidate_hidden = hidden_states  # Approximation
                    
                    # Calculate similarity with past tokens
                    similarities = torch.cosine_similarity(
                        candidate_hidden.unsqueeze(0),
                        torch.stack(past_hidden_states),
                        dim=-1
                    )
                    degeneration_penalty = torch.max(similarities)
                else:
                    degeneration_penalty = 0
                
                score = (1 - alpha) * model_score + alpha * degeneration_penalty
                scores.append(score)
            
            # Select best candidate
            best_idx = torch.argmax(torch.tensor(scores))
            next_token = top_k_indices[0][best_idx].unsqueeze(0).unsqueeze(0)
            
            generated = torch.cat([generated, next_token], dim=1)
            past_hidden_states.append(hidden_states)
    
    return generated
```

**Theoretical Properties**:
- **Fluency preservation**: Model probabilities ensure grammatical coherence
- **Diversity enhancement**: Similarity penalty reduces repetition
- **Controllable trade-off**: $\alpha$ parameter balances objectives

### Typical Sampling

**Information-Theoretic Motivation**
Select tokens with "typical" information content:
$$\text{Typical Set} = \{w : |\log p(w) - H(X)| < \epsilon\}$$

where $H(X)$ is the entropy of the distribution.

**Implementation**:
```python
def typical_sampling(logits, tau=0.95, temperature=1.0):
    # Apply temperature
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    
    # Calculate entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    
    # Calculate information content for each token
    info_content = -torch.log(probs)
    
    # Find tokens with typical information content
    deviation = torch.abs(info_content - entropy)
    
    # Sort by deviation from entropy
    sorted_deviations, sorted_indices = torch.sort(deviation)
    
    # Select tokens until cumulative mass reaches tau
    cumulative_mass = 0
    typical_indices = []
    for i, idx in enumerate(sorted_indices):
        cumulative_mass += probs[idx]
        typical_indices.append(idx)
        if cumulative_mass >= tau:
            break
    
    # Create filtered distribution
    filtered_probs = torch.zeros_like(probs)
    filtered_probs[typical_indices] = probs[typical_indices]
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    return torch.multinomial(filtered_probs, 1)
```

**Advantages**:
- **Information balance**: Avoids both too-obvious and too-surprising tokens
- **Adaptive**: Automatically adjusts to distribution characteristics
- **Theoretically grounded**: Based on information theory principles

## Controllable Text Generation

### Prompt-Based Control

**Instruction Following**
Modern language models can follow complex instructions:
```
Instruction: Write a haiku about artificial intelligence that includes the word "neural" and has a melancholic tone.

Response:
Neural pathways dark,
Silent thoughts in silicon—
Dreams we cannot share.
```

**Mathematical Model**:
$$P(\text{output}|\text{instruction}, \text{context}) = \frac{P(\text{output}|\text{context}) \cdot P(\text{instruction}|\text{output})}{P(\text{instruction}|\text{context})}$$

This represents Bayesian inference where we want outputs that:
1. Are probable given the context
2. Would likely generate the given instruction
3. Normalized by instruction probability

### Classifier-Free Guidance

**Principle**
Control generation using classifier gradients without explicit classifier training:
$$\nabla_x \log p(y|x) \approx \nabla_x \log p(x|y) + \nabla_x \log p(x)$$

**Implementation for Attribute Control**:
```python
def classifier_free_guidance(model, prompt, attribute_prompt, guidance_scale=7.5):
    """
    Generate text with classifier-free guidance
    
    Args:
        model: Language model
        prompt: Base prompt
        attribute_prompt: Prompt with desired attribute
        guidance_scale: Strength of guidance
    """
    # Get logits for both conditions
    with torch.no_grad():
        # Unconditional generation
        uncond_outputs = model(prompt)
        uncond_logits = uncond_outputs.logits
        
        # Conditional generation
        cond_outputs = model(attribute_prompt)
        cond_logits = cond_outputs.logits
        
        # Apply classifier-free guidance
        guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
        
        return guided_logits
```

**Steering Vector Approach**:
$$\mathbf{v}_{\text{steering}} = \mathbf{h}_{\text{desired}} - \mathbf{h}_{\text{neutral}}$$

where $\mathbf{h}$ represents hidden states for desired vs neutral conditions.

### CTRL (Conditional Transformer Language Model)

**Architecture Enhancement**
CTRL adds control codes to the input:
$$P(x_{t+1} | \mathbf{c}, x_1, ..., x_t)$$

where $\mathbf{c}$ is a control code specifying desired attributes.

**Control Code Types**:
- **Domain**: `Books`, `Wikipedia`, `News`
- **Style**: `Formal`, `Casual`, `Technical`
- **Length**: `Short`, `Medium`, `Long`
- **Topic**: `Politics`, `Science`, `Sports`

**Training Objective**:
$$\mathcal{L} = -\mathbb{E}_{(\mathbf{c}, \mathbf{x}) \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \log P(x_t | \mathbf{c}, x_{<t}) \right]$$

**Multi-Attribute Control**:
```python
def ctrl_generate(model, control_codes, prompt, max_length=100):
    """
    Generate text with multiple control codes
    
    control_codes: List of control attributes
    Example: ['Books', 'Formal', 'Long']
    """
    # Construct control sequence
    control_sequence = ' '.join(f'<{code}>' for code in control_codes)
    full_prompt = control_sequence + ' ' + prompt
    
    return model.generate(
        full_prompt,
        max_length=max_length,
        do_sample=True,
        temperature=0.8
    )
```

### Plug and Play Language Models (PPLM)

**Concept**
Combine a pretrained language model with small attribute classifiers:
$$P(x_t | x_{<t}, a) \propto P(x_t | x_{<t}) \cdot P(a | x_{\leq t})^{\gamma}$$

where $a$ is the desired attribute and $\gamma$ controls influence strength.

**Gradient-Based Steering**:
```python
def pplm_generate(language_model, classifier, attribute, prompt, steps=10, stepsize=0.02):
    input_ids = tokenize(prompt)
    
    for _ in range(steps):
        # Forward pass through language model
        outputs = language_model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Calculate attribute probability
        attr_logits = classifier(hidden_states.mean(dim=1))  # Pool over sequence
        attr_loss = -torch.log_softmax(attr_logits, dim=-1)[0, attribute]
        
        # Calculate gradients with respect to hidden states
        attr_loss.backward(retain_graph=True)
        
        # Update hidden states
        hidden_states_grad = hidden_states.grad
        hidden_states = hidden_states - stepsize * hidden_states_grad
        
        # Continue generation with modified representations
        next_token_logits = language_model.lm_head(hidden_states[:, -1, :])
        next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids
```

**Multi-Attribute Optimization**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fluency}} + \sum_{i} \lambda_i \mathcal{L}_{\text{attribute}_i}$$

### Constitutional AI

**Principle**
Train models to follow a set of principles or "constitution":
```
Constitutional Principles:
1. Be helpful and harmless
2. Respect human autonomy
3. Be truthful and admit uncertainty
4. Avoid generating harmful content
5. Respect privacy and confidentiality
```

**Training Process**:
1. **Constitutional Instruction**: Train model to follow principles
2. **Constitutional Feedback**: Use constitutional AI as evaluator
3. **Iterative Refinement**: Improve based on constitutional violations

**Mathematical Framework**:
$$P_{\text{constitutional}}(y|x) = P(y|x) \cdot \prod_{i} C_i(y)$$

where $C_i(y)$ are constitutional constraint functions.

**Implementation**:
```python
def constitutional_sampling(model, prompt, constitution, beta=2.0):
    """
    Sample text following constitutional principles
    """
    # Generate multiple candidates
    candidates = []
    for _ in range(10):
        candidate = model.generate(prompt, do_sample=True, temperature=0.8)
        candidates.append(candidate)
    
    # Evaluate each candidate against constitution
    scores = []
    for candidate in candidates:
        score = model.probability(candidate, given=prompt)
        
        # Apply constitutional constraints
        for principle in constitution:
            constraint_score = evaluate_principle(candidate, principle)
            score *= (constraint_score ** beta)
        
        scores.append(score)
    
    # Select best candidate
    best_idx = np.argmax(scores)
    return candidates[best_idx]
```

## Advanced Decoding Algorithms

### Beam Search Variations

**Standard Beam Search Limitations**:
- **Length bias**: Favors shorter sequences
- **Lack of diversity**: Multiple beams often collapse to similar sequences
- **Computational cost**: Exponential with beam width

**Length-Normalized Beam Search**:
$$\text{Score} = \frac{1}{|Y|^{\alpha}} \sum_{t=1}^{|Y|} \log P(y_t | y_{<t}, x)$$

where $\alpha$ controls length penalty strength.

**Diverse Beam Search**:
Maintain diversity among beams by penalizing similarity:
$$\text{Score}_{\text{diverse}} = \text{Score}_{\text{original}} - \lambda \cdot \text{Similarity}(\text{beam}_i, \text{beam}_j)$$

**Implementation**:
```python
def diverse_beam_search(model, input_ids, num_beams=5, diversity_penalty=0.5, num_groups=2):
    """
    Beam search with diversity penalty between groups
    """
    batch_size = input_ids.size(0)
    vocab_size = model.config.vocab_size
    
    # Initialize beams
    beam_scores = torch.zeros((batch_size, num_beams))
    beam_tokens = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
    
    # Divide beams into groups
    beams_per_group = num_beams // num_groups
    
    for step in range(max_length):
        # Get next token probabilities
        outputs = model(beam_tokens.view(-1, beam_tokens.size(-1)))
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
        
        # Reshape for beam search
        next_token_scores = next_token_scores.view(batch_size, num_beams, vocab_size)
        
        # Apply diversity penalty between groups
        for group_id in range(num_groups):
            group_start = group_id * beams_per_group
            group_end = (group_id + 1) * beams_per_group
            
            if group_id > 0:
                # Penalize tokens selected by previous groups
                for prev_group in range(group_id):
                    prev_start = prev_group * beams_per_group
                    prev_end = (prev_group + 1) * beams_per_group
                    
                    # Find tokens selected by previous group
                    prev_tokens = beam_tokens[:, prev_start:prev_end, -1]
                    
                    # Apply penalty
                    for token in prev_tokens.flatten():
                        next_token_scores[:, group_start:group_end, token] -= diversity_penalty
        
        # Select top beams
        candidate_scores = beam_scores.unsqueeze(-1) + next_token_scores
        candidate_scores = candidate_scores.view(batch_size, -1)
        
        top_scores, top_indices = torch.topk(candidate_scores, num_beams)
        
        # Update beams
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size
        
        beam_tokens = torch.cat([
            beam_tokens[torch.arange(batch_size).unsqueeze(1), beam_indices],
            token_indices.unsqueeze(-1)
        ], dim=-1)
        
        beam_scores = top_scores
    
    return beam_tokens, beam_scores
```

### Minimum Bayes Risk (MBR) Decoding

**Concept**
Select the hypothesis that minimizes expected risk:
$$\hat{y} = \arg\min_{y \in \mathcal{Y}} \mathbb{E}_{y' \sim P(y'|x)}[\mathcal{L}(y, y')]$$

where $\mathcal{L}(y, y')$ is a loss function between hypotheses.

**Implementation**:
```python
def mbr_decoding(model, input_ids, num_samples=100, num_candidates=10):
    """
    Minimum Bayes Risk decoding
    """
    # Generate multiple samples
    samples = []
    for _ in range(num_samples):
        sample = model.generate(input_ids, do_sample=True, temperature=0.8)
        samples.append(sample)
    
    # Select top candidates by probability
    candidates = []
    for _ in range(num_candidates):
        candidate = model.generate(input_ids, do_sample=True, temperature=0.6)
        candidates.append(candidate)
    
    # Calculate MBR scores
    mbr_scores = []
    for candidate in candidates:
        risk = 0
        for sample in samples:
            # Calculate similarity (e.g., BLEU, ROUGE, etc.)
            risk += 1 - calculate_similarity(candidate, sample)
        risk /= len(samples)
        mbr_scores.append(risk)
    
    # Select candidate with minimum risk
    best_idx = np.argmin(mbr_scores)
    return candidates[best_idx]
```

**Risk Functions**:
- **BLEU-based**: $\mathcal{L}(y, y') = 1 - \text{BLEU}(y, y')$
- **ROUGE-based**: $\mathcal{L}(y, y') = 1 - \text{ROUGE}(y, y')$
- **Semantic similarity**: $\mathcal{L}(y, y') = 1 - \cos(\text{embed}(y), \text{embed}(y'))$

### Speculative Decoding

**Motivation**
Accelerate autoregressive generation using a smaller "draft" model:
1. **Draft phase**: Small model generates multiple tokens quickly
2. **Verification phase**: Large model validates draft tokens
3. **Accept/Reject**: Accept valid tokens, reject and resample invalid ones

**Algorithm**:
```python
def speculative_decoding(large_model, small_model, input_ids, num_speculative_tokens=4):
    """
    Speculative decoding for faster generation
    """
    generated_ids = input_ids.clone()
    
    while len(generated_ids[0]) < max_length:
        # Draft phase: small model generates multiple tokens
        draft_ids = small_model.generate(
            generated_ids,
            max_new_tokens=num_speculative_tokens,
            do_sample=True,
            temperature=1.0
        )
        
        # Verification phase: large model evaluates draft
        with torch.no_grad():
            large_outputs = large_model(draft_ids)
            small_outputs = small_model(draft_ids)
            
            large_probs = torch.softmax(large_outputs.logits, dim=-1)
            small_probs = torch.softmax(small_outputs.logits, dim=-1)
        
        # Accept/reject tokens
        accepted_tokens = []
        for i in range(num_speculative_tokens):
            token_pos = len(generated_ids[0]) + i
            if token_pos >= len(draft_ids[0]):
                break
                
            draft_token = draft_ids[0, token_pos]
            
            # Acceptance probability
            r = large_probs[0, token_pos-1, draft_token] / small_probs[0, token_pos-1, draft_token]
            accept_prob = min(1.0, r)
            
            if torch.rand(1) < accept_prob:
                accepted_tokens.append(draft_token)
            else:
                # Reject and sample from modified distribution
                adjusted_probs = torch.max(
                    torch.zeros_like(large_probs[0, token_pos-1]),
                    large_probs[0, token_pos-1] - small_probs[0, token_pos-1]
                )
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                
                rejected_token = torch.multinomial(adjusted_probs, 1)
                accepted_tokens.append(rejected_token.item())
                break
        
        # Update generated sequence
        if accepted_tokens:
            new_tokens = torch.tensor([accepted_tokens]).to(generated_ids.device)
            generated_ids = torch.cat([generated_ids, new_tokens], dim=1)
    
    return generated_ids
```

**Speedup Analysis**:
$$\text{Speedup} = \frac{T_{\text{autoregressive}}}{T_{\text{speculative}}} = \frac{N \cdot T_{\text{large}}}{N/k \cdot (T_{\text{small}} \cdot k + T_{\text{large}})}$$

where $k$ is average number of accepted tokens per iteration.

## Multi-Objective Generation

### Pareto-Optimal Generation

**Problem Formulation**
Optimize multiple objectives simultaneously:
$$\max_{y} \{f_1(y), f_2(y), ..., f_k(y)\}$$

**Pareto Front Approximation**:
```python
def pareto_optimal_generation(model, input_ids, objectives, num_candidates=1000):
    """
    Generate Pareto-optimal candidates for multiple objectives
    """
    candidates = []
    objective_scores = []
    
    # Generate multiple candidates
    for _ in range(num_candidates):
        candidate = model.generate(input_ids, do_sample=True, temperature=0.8)
        candidates.append(candidate)
        
        # Evaluate all objectives
        scores = []
        for objective_fn in objectives:
            score = objective_fn(candidate)
            scores.append(score)
        objective_scores.append(scores)
    
    # Find Pareto front
    pareto_optimal_indices = []
    for i, scores_i in enumerate(objective_scores):
        is_dominated = False
        for j, scores_j in enumerate(objective_scores):
            if i != j and all(s_j >= s_i for s_j, s_i in zip(scores_j, scores_i)) and \
               any(s_j > s_i for s_j, s_i in zip(scores_j, scores_i)):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_optimal_indices.append(i)
    
    return [candidates[i] for i in pareto_optimal_indices]
```

**Multi-Objective Loss**:
$$\mathcal{L}_{MO} = \sum_{i=1}^{k} w_i \mathcal{L}_i$$

where weights $w_i$ can be:
- **Fixed**: Predetermined importance
- **Adaptive**: Learned during training
- **Dynamic**: Changed based on performance

### Quality-Diversity Trade-offs

**Novelty Search**
Encourage exploration of diverse solutions:
$$\text{Novelty}(y) = \frac{1}{k} \sum_{i=1}^{k} d(y, y_i^{\text{nearest}})$$

where $d(y, y_i)$ is distance to $k$-nearest neighbors.

**Quality-Diversity Objective**:
$$\text{Score}(y) = \alpha \cdot \text{Quality}(y) + (1-\alpha) \cdot \text{Novelty}(y)$$

**Implementation**:
```python
def quality_diversity_sampling(model, input_ids, alpha=0.7, num_candidates=50):
    """
    Balance quality and diversity in generation
    """
    candidates = []
    embeddings = []
    
    # Generate candidates
    for _ in range(num_candidates):
        candidate = model.generate(input_ids, do_sample=True, temperature=0.9)
        candidates.append(candidate)
        
        # Get embedding for novelty calculation
        embedding = get_text_embedding(candidate)
        embeddings.append(embedding)
    
    # Calculate quality scores
    quality_scores = [calculate_quality(candidate) for candidate in candidates]
    
    # Calculate novelty scores
    novelty_scores = []
    for i, embedding_i in enumerate(embeddings):
        distances = []
        for j, embedding_j in enumerate(embeddings):
            if i != j:
                distance = torch.cosine_similarity(embedding_i, embedding_j, dim=0)
                distances.append(1 - distance)  # Convert similarity to distance
        
        # Average distance to k nearest neighbors
        k_nearest = sorted(distances)[:5]  # k=5
        novelty = np.mean(k_nearest) if k_nearest else 0
        novelty_scores.append(novelty)
    
    # Combine scores
    combined_scores = [
        alpha * quality + (1 - alpha) * novelty
        for quality, novelty in zip(quality_scores, novelty_scores)
    ]
    
    # Select best candidate
    best_idx = np.argmax(combined_scores)
    return candidates[best_idx]
```

## Evaluation Methodologies

### Automatic Evaluation Metrics

**Perplexity-Based Metrics**
**Standard Perplexity**:
$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i})\right)$$

**Conditional Perplexity** (for controllable generation):
$$\text{PPL}_{\text{cond}} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i}, c)\right)$$

where $c$ is the control condition.

**MAUVE (Measuring Text Generation Diversity)**
Measures similarity between generated and human text distributions:
$$\text{MAUVE} = \int_0^1 p_{\text{curve}}(t) \cdot \log \frac{p_{\text{curve}}(t)}{r_{\text{curve}}(t)} dt$$

where $p_{\text{curve}}$ and $r_{\text{curve}}$ are probability curves of human and generated text.

**BERTScore**
Semantic similarity using contextual embeddings:
$$\text{BERTScore} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} \frac{\mathbf{x}_i^T \hat{\mathbf{x}}_j}{||\mathbf{x}_i|| \cdot ||\hat{\mathbf{x}}_j||}$$

### Human Evaluation Frameworks

**Multi-Dimensional Assessment**
Evaluate generation across multiple criteria:

```python
class GenerationEvaluator:
    def __init__(self):
        self.criteria = [
            'fluency',
            'coherence', 
            'relevance',
            'creativity',
            'factual_accuracy',
            'style_appropriateness'
        ]
    
    def evaluate_sample(self, generated_text, reference_text=None, context=None):
        scores = {}
        
        for criterion in self.criteria:
            score = self.evaluate_criterion(generated_text, criterion, reference_text, context)
            scores[criterion] = score
        
        return scores
    
    def evaluate_criterion(self, text, criterion, reference, context):
        # Implement criterion-specific evaluation
        if criterion == 'fluency':
            return self.evaluate_fluency(text)
        elif criterion == 'coherence':
            return self.evaluate_coherence(text, context)
        elif criterion == 'relevance':
            return self.evaluate_relevance(text, context)
        # ... other criteria
        
    def aggregate_scores(self, scores_list, method='weighted_average'):
        if method == 'weighted_average':
            weights = {
                'fluency': 0.2,
                'coherence': 0.2,
                'relevance': 0.25,
                'creativity': 0.15,
                'factual_accuracy': 0.15,
                'style_appropriateness': 0.05
            }
            
            aggregated = {}
            for criterion in self.criteria:
                criterion_scores = [scores[criterion] for scores in scores_list]
                aggregated[criterion] = np.mean(criterion_scores)
            
            overall_score = sum(
                aggregated[criterion] * weights[criterion] 
                for criterion in self.criteria
            )
            
            aggregated['overall'] = overall_score
            return aggregated
```

**Pairwise Comparison**
More reliable than absolute scoring:
$$P(\text{A} \succ \text{B}) = \frac{1}{1 + \exp(-\theta(\text{Quality}(\text{A}) - \text{Quality}(\text{B})))}$$

**Statistical Significance Testing**:
```python
def compare_generation_systems(system_a_scores, system_b_scores, alpha=0.05):
    """
    Statistical comparison of two generation systems
    """
    from scipy import stats
    
    # Paired t-test for overall scores
    t_stat, p_value = stats.ttest_rel(system_a_scores, system_b_scores)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(system_a_scores) - 1) * np.var(system_a_scores) + 
                         (len(system_b_scores) - 1) * np.var(system_b_scores)) / 
                        (len(system_a_scores) + len(system_b_scores) - 2))
    
    cohens_d = (np.mean(system_a_scores) - np.mean(system_b_scores)) / pooled_std
    
    # Bootstrap confidence interval
    def bootstrap_mean_diff(a, b, n_bootstrap=1000):
        diffs = []
        for _ in range(n_bootstrap):
            a_sample = np.random.choice(a, len(a), replace=True)
            b_sample = np.random.choice(b, len(b), replace=True)
            diffs.append(np.mean(a_sample) - np.mean(b_sample))
        return np.percentile(diffs, [2.5, 97.5])
    
    ci_lower, ci_upper = bootstrap_mean_diff(system_a_scores, system_b_scores)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': cohens_d,
        'confidence_interval': (ci_lower, ci_upper)
    }
```

### Task-Specific Evaluation

**Dialogue Generation**
Evaluate conversation quality:
- **Appropriateness**: Response relevance to context
- **Informativeness**: Amount of useful information provided
- **Engagement**: Ability to maintain interesting conversation
- **Consistency**: Maintaining consistent persona/knowledge

**Creative Writing**
Assess creative content:
- **Originality**: Novelty of ideas and expressions
- **Narrative structure**: Plot coherence and development
- **Character development**: Depth and consistency of characters
- **Emotional impact**: Ability to evoke emotions

**Code Generation**
Evaluate programming outputs:
- **Functional correctness**: Does code execute properly?
- **Efficiency**: Algorithm and space complexity
- **Readability**: Code style and documentation
- **Robustness**: Error handling and edge cases

**Factual Question Answering**:
```python
def evaluate_factual_qa(generated_answer, reference_answer, question_type):
    """
    Evaluate factual question answering
    """
    metrics = {}
    
    # Exact match
    metrics['exact_match'] = (generated_answer.strip().lower() == 
                             reference_answer.strip().lower())
    
    # Partial credit for numerical answers
    if question_type == 'numerical':
        try:
            gen_num = extract_number(generated_answer)
            ref_num = extract_number(reference_answer)
            
            if gen_num is not None and ref_num is not None:
                relative_error = abs(gen_num - ref_num) / abs(ref_num)
                metrics['numerical_accuracy'] = max(0, 1 - relative_error)
        except:
            metrics['numerical_accuracy'] = 0
    
    # Semantic similarity
    metrics['semantic_similarity'] = calculate_semantic_similarity(
        generated_answer, reference_answer
    )
    
    # Factual consistency check
    metrics['factual_consistency'] = check_factual_consistency(
        generated_answer, reference_answer
    )
    
    return metrics
```

## Production Deployment Strategies

### Real-Time Generation Systems

**Latency Optimization**
Minimize response time for interactive applications:

```python
class OptimizedGenerator:
    def __init__(self, model, cache_size=1000):
        self.model = model
        self.kv_cache = {}
        self.cache_size = cache_size
        self.response_cache = LRUCache(cache_size)
        
    def generate_streaming(self, prompt, max_tokens=100):
        """
        Stream tokens as they are generated
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Check cache for similar prompts
        cache_key = hash(prompt)
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            for token in cached_response:
                yield token
            return
        
        generated_tokens = []
        past_key_values = None
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Sample next token
                next_token = self.sample_token(logits)
                generated_tokens.append(next_token.item())
                
                # Yield immediately for streaming
                token_text = self.tokenizer.decode([next_token.item()])
                yield token_text
                
                # Prepare for next iteration
                input_ids = next_token.unsqueeze(0)
                
                # Stop on end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Cache the response
        self.response_cache[cache_key] = generated_tokens
    
    def sample_token(self, logits, temperature=0.8, top_p=0.9):
        """Optimized sampling"""
        if temperature == 0:
            return torch.argmax(logits, dim=-1)
        
        logits = logits / temperature
        
        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return torch.multinomial(F.softmax(logits, dim=-1), 1)
```

**Batch Processing**
Handle multiple requests efficiently:

```python
class BatchedGenerator:
    def __init__(self, model, max_batch_size=8):
        self.model = model
        self.max_batch_size = max_batch_size
        self.request_queue = []
        self.processing = False
        
    async def generate_batch(self, requests):
        """
        Process multiple generation requests in a batch
        """
        batch_size = min(len(requests), self.max_batch_size)
        batch_requests = requests[:batch_size]
        
        # Prepare batch input
        prompts = [req['prompt'] for req in batch_requests]
        max_lengths = [req.get('max_length', 100) for req in batch_requests]
        
        # Tokenize and pad
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Generate for batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max(max_lengths),
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode results
        results = []
        for i, (request, output) in enumerate(zip(batch_requests, outputs)):
            generated_text = self.tokenizer.decode(
                output[len(inputs['input_ids'][i]):],
                skip_special_tokens=True
            )
            
            results.append({
                'request_id': request['id'],
                'generated_text': generated_text,
                'prompt': request['prompt']
            })
        
        return results
    
    async def add_request(self, request):
        """Add request to processing queue"""
        self.request_queue.append(request)
        
        if not self.processing:
            await self.process_queue()
    
    async def process_queue(self):
        """Process queued requests in batches"""
        self.processing = True
        
        while self.request_queue:
            batch_requests = self.request_queue[:self.max_batch_size]
            self.request_queue = self.request_queue[self.max_batch_size:]
            
            results = await self.generate_batch(batch_requests)
            
            # Send results back to clients
            for result in results:
                await self.send_result(result)
        
        self.processing = False
```

### Quality Assurance and Monitoring

**Content Filtering**
Implement safety measures:

```python
class ContentFilter:
    def __init__(self):
        self.toxic_classifier = load_toxicity_model()
        self.bias_detector = load_bias_model()
        self.fact_checker = load_fact_checker()
        
    def filter_content(self, text, strict=False):
        """
        Comprehensive content filtering
        """
        results = {
            'approved': True,
            'issues': [],
            'modified_text': text
        }
        
        # Toxicity check
        toxicity_score = self.toxic_classifier.predict(text)
        if toxicity_score > (0.3 if strict else 0.7):
            results['approved'] = False
            results['issues'].append({
                'type': 'toxicity',
                'score': toxicity_score,
                'severity': 'high' if toxicity_score > 0.8 else 'medium'
            })
        
        # Bias detection
        bias_results = self.bias_detector.analyze(text)
        if any(score > 0.6 for score in bias_results.values()):
            results['approved'] = False
            results['issues'].append({
                'type': 'bias',
                'details': bias_results
            })
        
        # Fact checking (for factual claims)
        if self.contains_factual_claims(text):
            fact_check_results = self.fact_checker.verify(text)
            if fact_check_results['accuracy'] < 0.8:
                if strict:
                    results['approved'] = False
                results['issues'].append({
                    'type': 'factual_accuracy',
                    'accuracy': fact_check_results['accuracy'],
                    'disputed_claims': fact_check_results['disputed_claims']
                })
        
        # Apply corrections if not approved but correctable
        if not results['approved'] and self.can_correct(results['issues']):
            corrected_text = self.apply_corrections(text, results['issues'])
            results['modified_text'] = corrected_text
            results['approved'] = True
            results['corrected'] = True
        
        return results
    
    def contains_factual_claims(self, text):
        """Detect if text contains factual claims"""
        factual_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(?:studies show|research indicates|according to)\b',
            r'\b(?:percent|percentage|\%)\b',
            r'\b(?:scientists|researchers|experts) (?:found|discovered|concluded)\b'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in factual_patterns)
```

**Performance Monitoring**:
```python
class GenerationMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        
    def log_generation(self, request_id, prompt, generated_text, metadata):
        """Log generation event with metrics"""
        timestamp = time.time()
        
        # Calculate metrics
        latency = metadata.get('generation_time', 0)
        tokens_generated = len(self.tokenizer.encode(generated_text))
        tokens_per_second = tokens_generated / latency if latency > 0 else 0
        
        # Quality metrics
        perplexity = self.calculate_perplexity(generated_text)
        diversity = self.calculate_diversity(generated_text)
        
        # Log metrics
        self.metrics['latency'].append(latency)
        self.metrics['tokens_per_second'].append(tokens_per_second)
        self.metrics['perplexity'].append(perplexity)
        self.metrics['diversity'].append(diversity)
        
        # Check for anomalies
        self.check_anomalies(latency, perplexity, diversity)
        
        # Store generation record
        record = {
            'timestamp': timestamp,
            'request_id': request_id,
            'prompt_length': len(prompt),
            'generated_length': len(generated_text),
            'latency': latency,
            'tokens_per_second': tokens_per_second,
            'perplexity': perplexity,
            'diversity': diversity
        }
        
        self.store_record(record)
    
    def check_anomalies(self, latency, perplexity, diversity):
        """Detect performance anomalies"""
        # Latency alert
        if latency > 10.0:  # seconds
            self.alerts.append({
                'type': 'high_latency',
                'value': latency,
                'threshold': 10.0,
                'timestamp': time.time()
            })
        
        # Quality alerts
        if perplexity > 50:
            self.alerts.append({
                'type': 'low_quality',
                'metric': 'perplexity',
                'value': perplexity,
                'threshold': 50,
                'timestamp': time.time()
            })
        
        if diversity < 0.3:
            self.alerts.append({
                'type': 'low_diversity',
                'value': diversity,
                'threshold': 0.3,
                'timestamp': time.time()
            })
    
    def generate_report(self, time_window=3600):
        """Generate performance report"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Filter recent metrics
        recent_metrics = defaultdict(list)
        for metric_name, values in self.metrics.items():
            recent_metrics[metric_name] = [
                v for v, t in zip(values, range(len(values)))
                if current_time - t * 60 < time_window  # Approximate
            ]
        
        report = {}
        for metric_name, values in recent_metrics.items():
            if values:
                report[metric_name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'count': len(values)
                }
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.alerts
            if current_time - alert['timestamp'] < time_window
        ]
        
        report['alerts'] = recent_alerts
        report['alert_count'] = len(recent_alerts)
        
        return report
```

## Key Questions for Review

### Advanced Sampling
1. **Sampling Trade-offs**: How do different sampling strategies balance quality, diversity, and computational efficiency?

2. **Context-Aware Sampling**: How can sampling strategies adapt to different types of content and generation contexts?

3. **Multi-Objective Sampling**: What are effective methods for optimizing multiple generation objectives simultaneously?

### Controllable Generation
4. **Control Precision**: How can we achieve fine-grained control over generated content while maintaining naturalness?

5. **Multi-Attribute Control**: What are the challenges and solutions for controlling multiple attributes simultaneously?

6. **Robustness**: How can controllable generation systems maintain reliability across diverse inputs and conditions?

### Evaluation Methodology
7. **Automatic vs Human Evaluation**: What are the limitations of automatic metrics and when is human evaluation necessary?

8. **Task-Specific Metrics**: How should evaluation frameworks adapt to different generation tasks and domains?

9. **Long-Term Quality**: How can we evaluate generation systems for sustained quality over long interactions?

### Production Deployment
10. **Scalability**: What are the key technical challenges in deploying generation systems at scale?

11. **Safety and Ethics**: How can production systems maintain safety and ethical standards while preserving utility?

12. **Performance Optimization**: What are the most effective techniques for optimizing generation speed and resource usage?

### Future Directions
13. **Emerging Techniques**: What new approaches show promise for improving generation quality and control?

14. **Integration Challenges**: How can advanced generation techniques be effectively combined and integrated?

15. **Evaluation Evolution**: How should evaluation methodologies evolve as generation capabilities advance?

## Conclusion

Advanced generation techniques represent the sophisticated culmination of research into controllable, high-quality, and diverse text generation, providing the tools and methodologies necessary to harness the full potential of large language models for complex real-world applications across domains from creative writing to technical documentation and interactive AI systems. This comprehensive exploration has established:

**Sampling Innovation**: Deep understanding of sophisticated sampling strategies including nucleus sampling, contrastive search, and typical sampling demonstrates how mathematical principles can guide the development of generation algorithms that balance quality, diversity, and computational efficiency.

**Controllable Generation**: Systematic coverage of prompt-based control, classifier-free guidance, CTRL, PPLM, and constitutional AI reveals how language models can be steered to produce content with specific attributes while maintaining naturalness and coherence.

**Advanced Decoding**: Analysis of beam search variations, minimum Bayes risk decoding, and speculative decoding provides optimization strategies for different generation scenarios, from creative applications requiring diversity to factual applications requiring accuracy.

**Multi-Objective Optimization**: Integration of Pareto-optimal generation and quality-diversity trade-offs shows how to balance competing objectives in text generation, enabling applications that require optimization across multiple dimensions simultaneously.

**Evaluation Frameworks**: Comprehensive coverage of automatic metrics, human evaluation methodologies, and task-specific assessment provides the foundation for reliable evaluation of generation systems across diverse applications and quality dimensions.

**Production Deployment**: Understanding of real-time systems, batch processing, quality assurance, and monitoring demonstrates the practical considerations necessary for deploying advanced generation techniques in production environments at scale.

Advanced generation techniques are crucial for modern NLP applications because:
- **Quality Control**: Enable precise control over generation quality and attributes while maintaining fluency and coherence
- **Application Versatility**: Provide tools for diverse applications from creative writing to factual question answering
- **Production Readiness**: Offer scalable solutions for deploying generation systems in real-world environments
- **Evaluation Rigor**: Establish systematic approaches for assessing generation quality across multiple dimensions
- **Future Foundation**: Create the methodological basis for next-generation controllable AI systems

The techniques and insights covered provide essential knowledge for implementing sophisticated text generation systems, conducting rigorous evaluation of generation quality, and developing applications that leverage the full capabilities of large language models. Understanding these advanced methods is crucial for pushing the boundaries of what automated text generation can achieve and for developing AI systems that can reliably produce high-quality, controlled, and diverse text outputs tailored to specific user needs and application requirements.