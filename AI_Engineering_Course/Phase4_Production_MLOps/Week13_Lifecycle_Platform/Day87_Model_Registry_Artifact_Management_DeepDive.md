# Day 66: Model Registry & Artifact Management
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Interacting with MLflow Model Registry

Script to register, version, and transition models.

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "Llama-3-FineTuned"

def register_new_version(run_id, artifact_path):
    """Register a model from a run."""
    # Create registered model if not exists
    try:
        client.create_registered_model(model_name)
    except:
        pass
    
    # Create version
    result = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/{artifact_path}",
        run_id=run_id
    )
    print(f"Registered version: {result.version}")
    return result.version

def transition_stage(version, stage):
    """Transition model to Staging/Production."""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )
    print(f"Transitioned v{version} to {stage}")

def get_production_model():
    """Get the current production model URI."""
    models = client.get_latest_versions(model_name, stages=["Production"])
    if not models:
        return None
    return models[0].source

# Usage
# v = register_new_version("run_123", "model")
# transition_stage(v, "Staging")
```

### 2. Automated Promotion Gate (CI/CD Script)

A script that runs tests and promotes if successful.

```python
import sys

def run_tests(model_uri):
    print(f"Testing model at {model_uri}...")
    
    # 1. Load Model
    try:
        model = mlflow.pytorch.load_model(model_uri)
    except Exception as e:
        print(f"Failed to load: {e}")
        return False
        
    # 2. Smoke Test (Dummy Input)
    try:
        dummy_input = ["Hello world"]
        output = model.predict(dummy_input)
        print("Smoke test passed.")
    except Exception as e:
        print(f"Smoke test failed: {e}")
        return False
        
    # 3. Accuracy Check (Mock)
    accuracy = 0.95 # In reality, run eval on test set
    threshold = 0.90
    
    if accuracy < threshold:
        print(f"Accuracy {accuracy} below threshold {threshold}")
        return False
        
    return True

def main(model_name, version):
    model_uri = f"models:/{model_name}/{version}"
    
    if run_tests(model_uri):
        print("Tests passed. Promoting to Production.")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
    else:
        print("Tests failed. Rejecting.")
        sys.exit(1)

# main("Llama-3-FineTuned", "5")
```

### 3. Safetensors Conversion

Converting a PyTorch `.bin` (pickle) model to `.safetensors`.

```python
import torch
from safetensors.torch import save_file
from transformers import AutoModel

def convert_to_safetensors(model_id):
    print(f"Loading {model_id}...")
    # Load from HF or local path
    model = AutoModel.from_pretrained(model_id)
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Save as safetensors
    output_path = "model.safetensors"
    save_file(state_dict, output_path)
    print(f"Saved to {output_path}")
    
    # Verify loading
    from safetensors.torch import load_file
    loaded_state = load_file(output_path)
    print("Verification successful.")

# convert_to_safetensors("bert-base-uncased")
```

### 4. S3 Artifact Storage Structure

Designing a clean storage layout.

```python
import boto3

s3 = boto3.client('s3')
BUCKET = "my-model-registry"

def upload_artifacts(model_name, version, local_dir):
    """
    Uploads:
    s3://bucket/models/{name}/{version}/model.safetensors
    s3://bucket/models/{name}/{version}/config.json
    s3://bucket/models/{name}/{version}/tokenizer.json
    """
    prefix = f"models/{model_name}/{version}"
    
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        s3_path = f"{prefix}/{filename}"
        
        print(f"Uploading {local_path} -> s3://{BUCKET}/{s3_path}")
        s3.upload_file(local_path, BUCKET, s3_path)

def generate_presigned_url(model_name, version):
    """Generate URL for inference server to download."""
    key = f"models/{model_name}/{version}/model.safetensors"
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET, 'Key': key},
        ExpiresIn=3600
    )
    return url
```

### 5. Model Signing (Security)

Signing a model file with a private key.

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization

def sign_model(model_path, private_key_path):
    # 1. Load Private Key
    with open(private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(), password=None
        )
        
    # 2. Hash Model File
    with open(model_path, "rb") as f:
        model_data = f.read()
        
    # 3. Sign Hash
    signature = private_key.sign(
        model_data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    # 4. Save Signature
    with open(model_path + ".sig", "wb") as f:
        f.write(signature)
    print("Model signed.")

def verify_model(model_path, public_key_path):
    # Load Public Key, Load Signature, Verify...
    pass
```
