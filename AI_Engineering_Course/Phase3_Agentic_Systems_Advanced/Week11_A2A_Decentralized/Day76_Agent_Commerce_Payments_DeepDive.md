# Day 76: Agent Commerce & Payments
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Micropayment Agent (Solana)

We will build an agent that pays for a joke.
**Dependencies:** `solana`

```python
from solana.rpc.api import Client
from solana.transaction import Transaction
from solana.system_program import TransferParams, transfer
from solana.keypair import Keypair
from solana.publickey import PublicKey

class WalletAgent:
    def __init__(self, secret_key_bytes):
        self.client = Client("https://api.devnet.solana.com")
        self.keypair = Keypair.from_secret_key(secret_key_bytes)
        print(f"Agent Address: {self.keypair.public_key}")

    def get_balance(self):
        resp = self.client.get_balance(self.keypair.public_key)
        return resp["result"]["value"] / 1e9 # SOL

    def pay(self, receiver_address, amount_sol):
        print(f"Paying {amount_sol} SOL to {receiver_address}...")
        
        # Build Instruction
        ix = transfer(
            TransferParams(
                from_pubkey=self.keypair.public_key,
                to_pubkey=PublicKey(receiver_address),
                lamports=int(amount_sol * 1e9)
            )
        )
        
        # Build Transaction
        tx = Transaction().add(ix)
        
        # Sign and Send
        resp = self.client.send_transaction(tx, self.keypair)
        print(f"Transaction Signature: {resp['result']}")
        return resp['result']

# Mock Service
class JokeService:
    def __init__(self):
        self.address = Keypair().public_key # Generate random address
        self.price = 0.001

    def get_joke(self, tx_signature):
        # Verify TX on chain (omitted for brevity)
        # if verify(tx_signature, self.price):
        return "Why did the AI cross the road? To optimize the path."

# Usage
# agent = WalletAgent(b"...")
# service = JokeService()
# tx = agent.pay(service.address, 0.001)
# print(service.get_joke(tx))
```

### L402 (Lightning for HTTP)

L402 is a standard for "Payment-Required" APIs.
1.  **Client:** `GET /api/premium-data`
2.  **Server:** `402 Payment Required`. Header: `WWW-Authenticate: L402 invoice="lnbc1..."`
3.  **Client:** Pays the Lightning Invoice. Gets a `Preimage` (Proof of Payment).
4.  **Client:** `GET /api/premium-data`. Header: `Authorization: L402 <Preimage>`
5.  **Server:** Verifies Preimage. Returns Data.

This enables **Pay-Per-Request** APIs without accounts or API keys.

### Streaming Payments (Superfluid)

On EVM chains, you can open a stream.
*   `createFlow(receiver, rate="1 USDC/hour")`
*   The balance updates every second.
*   The Agent can stop the flow anytime.
Ideal for renting compute or continuous inference.

### Summary

*   **Solana/L2s:** Good for one-off low-fee payments.
*   **Lightning/L402:** Good for high-frequency API metering.
*   **Streaming:** Good for continuous services.
The Agent doesn't need a bank account; it just needs a private key.
