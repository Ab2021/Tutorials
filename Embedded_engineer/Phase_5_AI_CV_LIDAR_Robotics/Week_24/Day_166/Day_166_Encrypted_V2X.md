# Day 166: Encrypted V2X (PKI/Certificates)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 24: Cybersecurity & Robustness

---

> **📝 Content Creator Instructions:**
> Who are you? And why should I brake for you?
> - **Focus:** Public Key Infrastructure (PKI), Elliptic Curve Cryptography (ECDSA), Pseudonym Certificates (Privacy), and Message Signing/Verification.
> - **Code:** A Python script `v2x_crypto.py` using `cryptography`. Generate a Root CA, issue a Vehicle Cert, sign a BSM (Basic Safety Message), and verify it using the Public Key. Demonstrate a "Man-In-The-Middle" failure.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the V2X Trust Model (SCMS: Security Credential Management System).
2.  **Generate** ECDSA Key Pairs (Public/Private).
3.  **Sign** a message using the Private Key and **Verify** it with the Public Key.
4.  **Implement** Privacy Logic: Rotate Pseudonym Certs every 5 minutes.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None. (Real life: HSM - Hardware Security Module).

### Software Environment
```bash
pip install cryptography
```

### Prior Knowledge
- Asymmetric Encryption (Alice & Bob).
- Hashing (SHA-256).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Trust Problem

If I broadcast "I am a Fire Truck, move over!", how do you know I'm not a script kiddie?
*   **Solution:** Digital Signatures.
*   **Root CA:** The Government (DOT) signs a certificate for the Fire Truck.
*   **Verification:** You check the signature against the DOT's Public Key (which is hardcoded in your car).

### 🔹 Part 2: ECDSA (Elliptic Curve Digital Signature Algorithm)

RSA is too heavy/slow for 10Hz V2X.
*   **NIST P-256:** Curve used for V2X.
*   **Signature Size:** Small (~64 bytes).
*   **Speed:** Fast verification.

### 🔹 Part 3: Privacy & Pseudonyms

If I use the same Cert forever, I can tracked across the city.
*   **Pseudonym Certificates (PC):** Valid for 1 week. Rotated every 5 minutes.
*   **Linkage Authority:** Checks if two PCs belong to the same car (Only for law enforcement).

---

## 💻 Implementation: The V2X PKI

We simulate a mini SCMS.

### 🛠️ Project Structure
```text
day166_crypto/
├── src/
│   ├── v2x_crypto.py
└── output/
    ├── crypto_test.txt
```

### 👨‍💻 V2X Signing(`src/v2x_crypto.py`)

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
import time
import json

class CertificateAuthority:
    def __init__(self, name):
        self.name = name
        # Generate Root Key
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()
        
    def get_public_pem(self):
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
    def sign_certificate(self, device_public_key, valid_until):
        # A Real Cert is simpler here: Just signing the Device's PubKey + Expiry
        # Data to sign: PubKey_Bytes + Expiry_Bytes
        
        device_pub_bytes = device_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        payload = device_pub_bytes + str(valid_until).encode('utf-8')
        
        signature = self.private_key.sign(
            payload,
            ec.ECDSA(hashes.SHA256())
        )
        
        return {'pub_key': device_pub_bytes.decode('utf-8'), 
                'valid_until': valid_until, 
                'ca_signature': signature.hex()}

class VehicleOBU:
    def __init__(self, id):
        self.id = id
        # Key for this Pseudonym
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()
        self.cert = None
        
    def get_certificate(self, ca):
        # Request cert valid for 60 seconds
        expiry = time.time() + 60
        self.cert = ca.sign_certificate(self.public_key, expiry)
        print(f"[{self.id}] Acquired Certificate from {ca.name}")
        
    def sign_bsm(self, message_dict):
        # Data: Position, Velocity, etc.
        data_bytes = json.dumps(message_dict, sort_keys=True).encode('utf-8')
        
        # Sign with private key
        signature = self.private_key.sign(
            data_bytes,
            ec.ECDSA(hashes.SHA256())
        )
        
        # Packet includes: Message + Signature + Certificate (to verify signature)
        packet = {
            'payload': message_dict,
            'signature': signature.hex(),
            'certificate': self.cert
        }
        return packet

class VehicleReceiver:
    def __init__(self, ca_public_pem):
        self.ca_public_key = serialization.load_pem_public_key(ca_public_pem)
        
    def verify_packet(self, packet):
        try:
            cert = packet['certificate']
            signature = bytes.fromhex(packet['signature'])
            payload = packet['payload']
            data_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
            
            # 1. Verify Certificate Chain (Is the Cert signed by CA?)
            cert_pub_pem = cert['pub_key'].encode('utf-8')
            cert_expiry = float(cert['valid_until'])
            ca_sig = bytes.fromhex(cert['ca_signature'])
            
            # Reconstruct CA signed data
            ca_payload = cert_pub_pem + str(cert['valid_until']).encode('utf-8')
            
            self.ca_public_key.verify(
                ca_sig,
                ca_payload,
                ec.ECDSA(hashes.SHA256())
            )
            print("   [Check 1/3] Certificate issued by Trusted CA: PASS")
            
            # 2. Check Expiry
            if time.time() > cert_expiry:
                print("   [Check 2/3] Certificate Expiry: FAIL (Expired)")
                return False
            print("   [Check 2/3] Certificate Expiry: PASS")
            
            # 3. Verify Message Signature using Cert's Public Key
            sender_pub_key = serialization.load_pem_public_key(cert_pub_pem)
            sender_pub_key.verify(
                signature,
                data_bytes,
                ec.ECDSA(hashes.SHA256())
            )
            print("   [Check 3/3] Message Signature: PASS")
            
            return True
            
        except InvalidSignature:
            print("   !!! SIG VERIFICATION FAILED !!!")
            return False
        except Exception as e:
            print(f"   Error: {e}")
            return False

def main():
    print("--- V2X PKI Simulation ---")
    
    # 1. Setup Infrastructure
    root_ca = CertificateAuthority("US_DOT_ROOT")
    
    # 2. Setup Car (Alice)
    alice = VehicleOBU("Alice_Car")
    alice.get_certificate(root_ca)
    
    # 3. Setup Receiver (Bob)
    bob = VehicleReceiver(root_ca.get_public_pem())
    
    # Case A: Valid Message
    print("\n[Case A] Alice sends valid BSM...")
    bsm = {'lat': 37.77, 'lon': -122.41, 'speed': 30}
    packet = alice.sign_bsm(bsm)
    result = bob.verify_packet(packet)
    print(f"Packet Authentic: {result}")
    
    # Case B: Tampering (Man-In-The-Middle)
    print("\n[Case B] Eve changes Speed to 0...")
    packet_bad = packet.copy()
    # Modify payload BUT keep Alice's signature
    packet_bad['payload'] = {'lat': 37.77, 'lon': -122.41, 'speed': 0} 
    result = bob.verify_packet(packet_bad)
    print(f"Packet Authentic: {result}")
    
    # Case C: Expired Cert
    # Manually expire Alice's cert
    print("\n[Case C] Alice uses old cert...")
    alice.cert['valid_until'] = time.time() - 10 # expired
    packet_expired = alice.sign_bsm(bsm)
    result = bob.verify_packet(packet_expired)
    print(f"Packet Authentic: {result}")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "CRL (Certificate Revocation List)"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Expiry check works.
- **Feature:** Revocation.
- **Modify:** `VehicleReceiver` keeps a list `CRL = [hash(hacked_cert_pubkey)]`.
- **Logic:** Before Check 2, calculate hash of `cert['pub_key']`. If in CRL, Reject.
- **Scenario:** Alice is reported as malicious. CA publishes CRL. Bob downloads CRL. Alice is blocked.

---

## 🚀 Project: "The Butterfly Effect"

**Goal:** Butterfly Key Expansion.
1.  **Concept:** Car downloads 3000 certs for the week? No.
2.  **Implementation:** Car downloads 1 Seed Key. Generates 3000 certs locally. CA can reconstruct them to verify.
3.  **Task:** Read about **IEEE 1609.2** Butterfly Keys. (Simulate minimal version: Deterministic Key Gen).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Slow Verification"
*   **Cause:** Python `cryptography` pure python fallback?
*   **Fix:** Ensure OpenSSL backend is used. ECDSA verify should take < 1ms.

#### 2. "Clock Skew"
*   **Cause:** Alice clock is 5 mins ahead. Bob rejects "Future" certs.
*   **Fix:** GPS Time Sync is mandatory for V2X.

---

## ⚡ Optimization: Hardware Security Module (HSM)

Private Keys never leave the chip.
*   **HSM:** Tamper-proof chip.
*   **CMD:** `HSM_Sign(Hash)` -> `Signature`.
*   **Benefit:** If hacker steals the OBU, they cannot extract the `private_key` to clone the car identity.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Shared Secret vs Public Key?
    *   **A:** Shared Secret (Symmetric) requires secure channel to share key. PKI (Asymmetric) allows public verification.
2.  **Q:** Why not RSA?
    *   **A:** Keys are too big (2048 bits vs 256 bits ECC) for same security. Bandwidth is precious.
3.  **Q:** What is a BSM?
    *   **A:** Basic Safety Message. Sent at 10Hz. contains Pos, Vel, Accel, Wheel Angle.

### Challenge Task
> **Task:** Certificate Rolling.
> 1. Issue a Batch of 5 certs to Alice.
> 2. Alice uses Cert 1 for min 0-1, Cert 2 for min 1-2.
> 3. Verify Privacy: Does Bob know Cert 1 and Cert 2 are the same car? (No, unless Linkage Authority).

---

## 📚 Further Reading
- **IEEE 1609.2:** Standard for Wireless Access in Vehicular Environments (WAVE) - Security Services.
- **CAMP:** V2V Security Credential Management System (SCMS).

---

**Day 166 Complete**
