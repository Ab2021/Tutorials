# Day 42: Multi-Agent Systems
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Debate System Implementation

```python
import openai
from typing import List, Dict

class DebateSystem:
    def __init__(self, topic: str):
        self.topic = topic
        self.history = []
    
    def run_debate(self, rounds: int = 3) -> Dict:
        """Run a debate between Pro and Con agents."""
        for round_num in range(rounds):
            # Pro argument
            pro_arg = self._get_argument("pro", round_num)
            self.history.append({"role": "pro", "content": pro_arg})
            
            # Con argument
            con_arg = self._get_argument("con", round_num)
            self.history.append({"role": "con", "content": con_arg})
        
        # Judge decides
        decision = self._judge()
        
        return {
            "topic": self.topic,
            "history": self.history,
            "decision": decision
        }
    
    def _get_argument(self, side: str, round_num: int) -> str:
        """Generate argument for one side."""
        # Build context from history
        context = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.history
        ])
        
        prompt = f"""You are arguing {side.upper()} for: {self.topic}

Previous arguments:
{context if context else 'None'}

Round {round_num + 1}: Present your argument. Be specific and persuasive.

Argument:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _judge(self) -> str:
        """Judge decides the winner."""
        debate_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.history
        ])
        
        prompt = f"""You are an impartial judge. Review this debate and decide which side presented stronger arguments.

Topic: {self.topic}

Debate:
{debate_text}

Decision (Pro/Con) with brief explanation:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content

# Usage
debate = DebateSystem("Should AI development be regulated?")
result = debate.run_debate(rounds=3)
print(result["decision"])
```

### 2. Ensemble System

```python
class EnsembleSystem:
    def __init__(self, num_agents: int = 5):
        self.num_agents = num_agents
    
    def query(self, question: str) -> str:
        """Query multiple agents and aggregate."""
        # Get answers from all agents
        answers = []
        for i in range(self.num_agents):
            answer = self._get_answer(question, agent_id=i)
            answers.append(answer)
        
        # Aggregate
        final_answer = self._aggregate(question, answers)
        
        return final_answer
    
    def _get_answer(self, question: str, agent_id: int) -> str:
        """Get answer from one agent."""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
            temperature=0.7 + (agent_id * 0.1)  # Vary temperature for diversity
        )
        return response.choices[0].message.content
    
    def _aggregate(self, question: str, answers: List[str]) -> str:
        """Aggregate multiple answers."""
        answers_text = "\n\n".join([
            f"Answer {i+1}: {ans}"
            for i, ans in enumerate(answers)
        ])
        
        prompt = f"""You have {len(answers)} different answers to a question. Synthesize them into one best answer.

Question: {question}

Answers:
{answers_text}

Synthesized Answer:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content
```

### 3. Hierarchical System (Manager-Worker)

```python
class HierarchicalSystem:
    def __init__(self):
        self.workers = {
            "researcher": self._research_worker,
            "coder": self._code_worker,
            "writer": self._write_worker
        }
    
    def execute_task(self, task: str) -> str:
        """Manager delegates to workers."""
        # Manager decomposes task
        subtasks = self._decompose_task(task)
        
        # Execute subtasks
        results = {}
        for subtask in subtasks:
            worker_type = subtask["worker"]
            worker_task = subtask["task"]
            
            if worker_type in self.workers:
                result = self.workers[worker_type](worker_task)
                results[worker_type] = result
        
        # Manager synthesizes
        final_output = self._synthesize(task, results)
        
        return final_output
    
    def _decompose_task(self, task: str) -> List[Dict]:
        """Manager decomposes task into subtasks."""
        prompt = f"""You are a manager. Decompose this task into subtasks for specialized workers.

Available workers: researcher, coder, writer

Task: {task}

Subtasks (JSON array):
[
  {{"worker": "researcher", "task": "..."}},
  {{"worker": "coder", "task": "..."}},
  ...
]

Subtasks:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    
    def _research_worker(self, task: str) -> str:
        """Research worker."""
        prompt = f"You are a researcher. {task}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def _code_worker(self, task: str) -> str:
        """Code worker."""
        prompt = f"You are a programmer. {task}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def _write_worker(self, task: str) -> str:
        """Writer worker."""
        prompt = f"You are a writer. {task}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def _synthesize(self, task: str, results: Dict) -> str:
        """Manager synthesizes results."""
        results_text = "\n\n".join([
            f"{worker.upper()}: {result}"
            for worker, result in results.items()
        ])
        
        prompt = f"""Synthesize these worker outputs into a final answer.

Task: {task}

Worker Outputs:
{results_text}

Final Answer:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

### 4. Reflection System (Generator + Critic)

```python
class ReflectionSystem:
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
    
    def generate_with_reflection(self, task: str) -> str:
        """Generate with iterative refinement."""
        output = self._generate(task)
        
        for iteration in range(self.max_iterations):
            # Critic reviews
            critique = self._critique(task, output)
            
            # Check if satisfactory
            if "satisfactory" in critique.lower():
                break
            
            # Revise based on critique
            output = self._revise(task, output, critique)
        
        return output
    
    def _generate(self, task: str) -> str:
        """Generator creates initial output."""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": task}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def _critique(self, task: str, output: str) -> str:
        """Critic reviews output."""
        prompt = f"""You are a critic. Review this output and provide constructive feedback.

Task: {task}

Output:
{output}

If the output is good, say "Satisfactory". Otherwise, provide specific improvements.

Critique:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    
    def _revise(self, task: str, output: str, critique: str) -> str:
        """Generator revises based on critique."""
        prompt = f"""Revise your output based on the critique.

Task: {task}

Original Output:
{output}

Critique:
{critique}

Revised Output:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
```
