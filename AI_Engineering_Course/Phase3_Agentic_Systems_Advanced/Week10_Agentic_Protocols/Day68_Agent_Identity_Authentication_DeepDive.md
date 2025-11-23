# Day 68: Agent Identity & Authentication
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a DID-Based Agent

We will simulate an agent signing a message using a simplified DID workflow.

**Dependencies:**
`pip install ecdsa`

**Code (`identity.py`):**

```python
import json
import time
from ecdsa import SigningKey, SECP256k1
import hashlib

class AgentIdentity:
    def __init__(self, name):
        self.name = name
        # Generate Key Pair (The "Wallet")
        self._sk = SigningKey.generate(curve=SECP256k1)
        self.vk = self._sk.verifying_key
        self.did = f"did:agent:{self.vk.to_string().hex()[:16]}"
        
    def sign_task(self, task_payload):
        """Sign a task to prove it came from this agent."""
        # 1. Canonicalize
        payload_str = json.dumps(task_payload, sort_keys=True)
        
        # 2. Sign
        signature = self._sk.sign(payload_str.encode())
        
        return {
            "payload": task_payload,
            "signature": signature.hex(),
            "signer": self.did,
            "public_key": self.vk.to_string().hex()
        }

class ServiceProvider:
    def verify_request(self, signed_request):
        """Verify the signature."""
        vk_hex = signed_request["public_key"]
        sig_hex = signed_request["signature"]
        payload = signed_request["payload"]
        
        # Reconstruct keys
        vk = VerifyingKey.from_string(bytes.fromhex(vk_hex), curve=SECP256k1)
        payload_str = json.dumps(payload, sort_keys=True)
        
        try:
            vk.verify(bytes.fromhex(sig_hex), payload_str.encode())
            print(f"✅ Verified request from {signed_request['signer']}")
            return True
        except:
            print("❌ Signature Invalid!")
            return False

# Usage
agent = AgentIdentity("ShoppingBot")
service = ServiceProvider()

request = {
    "action": "buy_item",
    "item_id": "123",
    "price": 50,
    "timestamp": time.time()
}

signed_req = agent.sign_task(request)
service.verify_request(signed_req)
```

### UCANs (User Controlled Authorization Networks)

UCANs are a modern token format (an evolution of JWT) designed for decentralized auth.
They allow **Chained Delegation**.

**Scenario:**
1.  Alice has root access to `s3://alice-bucket`.
2.  Alice issues a UCAN to `AgentA` with capability `s3:write` on `s3://alice-bucket/photos`.
3.  `AgentA` issues a UCAN to `SubAgentB` with capability `s3:write` on `s3://alice-bucket/photos/vacation`.

When `SubAgentB` writes to S3, it presents the *chain* of UCANs. S3 verifies:
*   SubAgentB was authorized by AgentA.
*   AgentA was authorized by Alice.
*   Alice owns the bucket.

This enables **Least Privilege** without Alice ever knowing SubAgentB exists.

### Implementing a UCAN-like Token (Conceptual)

```json
{
  "iss": "did:key:alice",
  "aud": "did:key:agent-a",
  "att": [
    {
      "with": "s3://alice-bucket/photos",
      "can": "write"
    }
  ],
  "prf": [] // Proofs (parent tokens)
}
```

### Summary

*   **Keys:** The Agent holds a private key.
*   **Signatures:** Every action is signed.
*   **Delegation:** Users sign tokens delegating specific rights to the Agent's key.
This creates a cryptographic chain of custody for every AI action.
