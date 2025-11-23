# Day 56: Evaluation & Benchmarking
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Complete Evaluation Pipeline

```python
import numpy as np
from typing import List, Dict
import openai
from rouge_score import rouge_scorer
from bert_score import score as bert_score

class LLMEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """Comprehensive evaluation."""
        results = {
            'rouge': self._compute_rouge(predictions, references),
            'bertscore': self._compute_bertscore(predictions, references),
            'length_stats': self._compute_length_stats(predictions)
        }
        
        return results
    
    def _compute_rouge(self, predictions, references):
        """Compute ROUGE scores."""
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            for key in scores:
                scores[key].append(score[key].fmeasure)
        
        # Average scores
        return {key: np.mean(vals) for key, vals in scores.items()}
    
    def _compute_bertscore(self, predictions, references):
        """Compute BERTScore."""
        P, R, F1 = bert_score(
            predictions,
            references,
            lang='en',
            verbose=False
        )
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    
    def _compute_length_stats(self, predictions):
        """Compute length statistics."""
        lengths = [len(pred.split()) for pred in predictions]
        
        return {
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths)
        }
```

### 2. LLM-as-Judge Implementation

```python
class LLMJudge:
    def __init__(self, judge_model='gpt-4'):
        self.judge_model = judge_model
    
    def evaluate_response(
        self,
        question: str,
        response: str,
        criteria: List[str] = None
    ) -> Dict:
        """Evaluate response using LLM judge."""
        if criteria is None:
            criteria = ['helpfulness', 'accuracy', 'clarity']
        
        # Create evaluation prompt
        prompt = self._create_eval_prompt(question, response, criteria)
        
        # Get judgment
        judgment = openai.ChatCompletion.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        # Parse scores
        scores = self._parse_judgment(judgment.choices[0].message.content)
        
        return scores
    
    def _create_eval_prompt(self, question, response, criteria):
        """Create evaluation prompt."""
        criteria_text = '\n'.join([f"- {c.capitalize()}" for c in criteria])
        
        prompt = f"""Evaluate the following response on a scale of 1-10 for each criterion.

Question: {question}

Response: {response}

Criteria:
{criteria_text}

Provide scores in the format:
Helpfulness: X/10
Accuracy: X/10
Clarity: X/10

Scores:"""
        
        return prompt
    
    def _parse_judgment(self, judgment_text):
        """Parse scores from judgment."""
        scores = {}
        
        for line in judgment_text.split('\n'):
            if ':' in line:
                criterion, score = line.split(':')
                criterion = criterion.strip().lower()
                score = score.strip().split('/')[0]
                try:
                    scores[criterion] = float(score)
                except:
                    pass
        
        return scores
    
    def pairwise_comparison(
        self,
        question: str,
        response_a: str,
        response_b: str
    ) -> str:
        """Compare two responses."""
        prompt = f"""Compare the following two responses to the question.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Which response is better? Answer with 'A', 'B', or 'Tie'.

Answer:"""
        
        judgment = openai.ChatCompletion.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        answer = judgment.choices[0].message.content.strip()
        
        return answer
```

### 3. MT-Bench Implementation

```python
class MTBenchEvaluator:
    def __init__(self, judge_model='gpt-4'):
        self.judge = LLMJudge(judge_model)
        self.categories = [
            'writing', 'roleplay', 'reasoning', 'math',
            'coding', 'extraction', 'stem', 'humanities'
        ]
    
    def evaluate_model(self, model, questions: List[Dict]) -> Dict:
        """Evaluate model on MT-Bench."""
        scores_by_category = {cat: [] for cat in self.categories}
        
        for q in questions:
            # First turn
            response_1 = model.generate(q['turns'][0])
            score_1 = self.judge.evaluate_response(
                q['turns'][0],
                response_1
            )
            
            # Second turn
            response_2 = model.generate(
                q['turns'][1],
                context=[q['turns'][0], response_1]
            )
            score_2 = self.judge.evaluate_response(
                q['turns'][1],
                response_2
            )
            
            # Average score for this question
            avg_score = (score_1['overall'] + score_2['overall']) / 2
            scores_by_category[q['category']].append(avg_score)
        
        # Compute category averages
        results = {
            cat: np.mean(scores) if scores else 0
            for cat, scores in scores_by_category.items()
        }
        
        # Overall score
        results['overall'] = np.mean(list(results.values()))
        
        return results
```

### 4. Human Evaluation Framework

```python
class HumanEvaluator:
    def __init__(self):
        self.annotations = []
    
    def collect_pairwise_annotation(
        self,
        question: str,
        response_a: str,
        response_b: str,
        annotator_id: str
    ):
        """Collect pairwise comparison from human."""
        print(f"\nQuestion: {question}")
        print(f"\nResponse A: {response_a}")
        print(f"\nResponse B: {response_b}")
        
        choice = input("\nWhich is better? (A/B/Tie): ").strip().upper()
        
        self.annotations.append({
            'question': question,
            'response_a': response_a,
            'response_b': response_b,
            'choice': choice,
            'annotator_id': annotator_id
        })
    
    def compute_win_rate(self, model_a_name: str, model_b_name: str):
        """Compute win rate for model A vs model B."""
        wins_a = sum(1 for a in self.annotations if a['choice'] == 'A')
        wins_b = sum(1 for a in self.annotations if a['choice'] == 'B')
        ties = sum(1 for a in self.annotations if a['choice'] == 'Tie')
        
        total = len(self.annotations)
        
        return {
            f'{model_a_name}_win_rate': wins_a / total,
            f'{model_b_name}_win_rate': wins_b / total,
            'tie_rate': ties / total
        }
    
    def compute_inter_annotator_agreement(self):
        """Compute agreement between annotators."""
        # Group by question
        by_question = {}
        for ann in self.annotations:
            q = ann['question']
            if q not in by_question:
                by_question[q] = []
            by_question[q].append(ann['choice'])
        
        # Compute agreement
        agreements = []
        for choices in by_question.values():
            if len(choices) >= 2:
                # Pairwise agreement
                agree = sum(
                    1 for i in range(len(choices))
                    for j in range(i+1, len(choices))
                    if choices[i] == choices[j]
                )
                total_pairs = len(choices) * (len(choices) - 1) / 2
                agreements.append(agree / total_pairs)
        
        return np.mean(agreements) if agreements else 0
```

### 5. Calibration Measurement

```python
class CalibrationEvaluator:
    def __init__(self, num_bins=10):
        self.num_bins = num_bins
    
    def compute_ece(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                # Accuracy in this bin
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                
                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Bin weight
                bin_weight = in_bin.sum() / len(confidences)
                
                # Add to ECE
                ece += bin_weight * abs(avg_confidence_in_bin - accuracy_in_bin)
        
        return ece
```

### 6. Benchmark Runner

```python
class BenchmarkRunner:
    def __init__(self, model):
        self.model = model
        self.results = {}
    
    def run_mmlu(self, dataset):
        """Run MMLU benchmark."""
        correct = 0
        total = 0
        
        for example in dataset:
            question = example['question']
            choices = example['choices']
            answer = example['answer']
            
            # Format prompt
            prompt = f"{question}\n\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "\nAnswer:"
            
            # Get prediction
            response = self.model.generate(prompt, max_tokens=1)
            prediction = response.strip()[0]
            
            # Check correctness
            if prediction == chr(65 + answer):
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.results['mmlu'] = accuracy
        
        return accuracy
    
    def run_humaneval(self, dataset):
        """Run HumanEval benchmark."""
        passed = 0
        total = 0
        
        for example in dataset:
            prompt = example['prompt']
            test_cases = example['test']
            
            # Generate code
            code = self.model.generate(prompt, max_tokens=512)
            
            # Run test cases
            try:
                exec(code + '\n' + test_cases)
                passed += 1
            except:
                pass
            
            total += 1
        
        pass_rate = passed / total
        self.results['humaneval'] = pass_rate
        
        return pass_rate
```
