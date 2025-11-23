# Day 67: Continuous Training (CT) Pipelines
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. CT Pipeline Implementation (Conceptual Airflow DAG)

Defining the workflow as code.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'llm_continuous_training',
    default_args=default_args,
    schedule_interval='@weekly',
    start_date=datetime(2023, 1, 1),
) as dag:

    def validate_data():
        # Check for drift, schema, volume
        print("Validating new data...")
        if data_volume < 1000:
            raise ValueError("Not enough data")

    def train_lora():
        # Run LoRA fine-tuning job
        print("Training LoRA adapter...")
        # cmd = "python train.py --config config.yaml"
        return "model_v2_candidate"

    def evaluate_model(ti):
        model_path = ti.xcom_pull(task_ids='train_lora')
        print(f"Evaluating {model_path}...")
        accuracy = 0.95
        if accuracy < 0.90:
            raise ValueError("Model degraded")
        return accuracy

    def register_model():
        print("Registering model to MLflow...")

    # Tasks
    t1 = PythonOperator(task_id='validate_data', python_callable=validate_data)
    t2 = PythonOperator(task_id='train_lora', python_callable=train_lora)
    t3 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model)
    t4 = PythonOperator(task_id='register_model', python_callable=register_model)

    t1 >> t2 >> t3 >> t4
```

### 2. Drift Detection Trigger

Detecting when to retrain based on embedding drift.

```python
import numpy as np
from scipy.spatial.distance import jensenshannon

def calculate_drift(ref_embeddings, curr_embeddings):
    """
    Calculate drift between reference (training) and current (production) data.
    Using simple distribution shift of embedding norms or PCA components.
    """
    # Simplified: Compare mean/std
    ref_mean = np.mean(ref_embeddings, axis=0)
    curr_mean = np.mean(curr_embeddings, axis=0)
    
    dist = np.linalg.norm(ref_mean - curr_mean)
    print(f"Drift Distance: {dist}")
    
    return dist

def check_trigger(drift_score, threshold=0.1):
    if drift_score > threshold:
        print("Drift detected! Triggering CT pipeline.")
        return True
    return False

# Usage
# ref = load_training_embeddings()
# curr = load_production_embeddings_last_24h()
# drift = calculate_drift(ref, curr)
# check_trigger(drift)
```

### 3. LoRA Fine-Tuning Script for CT

A lightweight training script suitable for automated jobs.

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

def run_ct_training(base_model_id, dataset):
    # 1. Load Base Model (Frozen)
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    
    # 2. Config LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    
    # 3. Trainer
    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1, # Fast training
        per_device_train_batch_size=4,
        learning_rate=2e-4
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )
    
    # 4. Train
    trainer.train()
    
    # 5. Save Adapter
    model.save_pretrained("./new_adapter")
    return "./new_adapter"
```

### 4. Shadow Deployment Logic

Simulating a shadow mode router.

```python
import threading

class ShadowRouter:
    def __init__(self, prod_model, shadow_model):
        self.prod = prod_model
        self.shadow = shadow_model
        
    def handle_request(self, input_text):
        # 1. Start Shadow Inference (Async)
        shadow_thread = threading.Thread(
            target=self._log_shadow_prediction,
            args=(input_text,)
        )
        shadow_thread.start()
        
        # 2. Return Prod Prediction (Sync)
        return self.prod.predict(input_text)
        
    def _log_shadow_prediction(self, text):
        pred = self.shadow.predict(text)
        # Log to DB for later comparison
        print(f"Shadow Prediction: {pred}")
        # Compare with Prod?
        
# Usage
# router = ShadowRouter(v1, v2)
# response = router.handle_request("Hello")
```

### 5. Replay Buffer Implementation

Mixing old data to prevent forgetting.

```python
import random

def create_replay_dataset(new_data, old_data, ratio=0.1):
    """
    Mix 10% old data into the new training set.
    """
    n_new = len(new_data)
    n_replay = int(n_new * ratio)
    
    replay_samples = random.sample(old_data, min(n_replay, len(old_data)))
    
    combined = new_data + replay_samples
    random.shuffle(combined)
    
    return combined
```
