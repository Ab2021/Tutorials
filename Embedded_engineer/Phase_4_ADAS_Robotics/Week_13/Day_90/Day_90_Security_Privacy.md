# Day 90: Security and Privacy (PKI)
## Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication

---

> **📝 Day 90 Focus:**
> If a hacker can spoof a "Red Light" message, they can stop traffic in a whole city. If they can track your BSMs, they know where you live. **V2X Security** ensures messages are authentic, and **Privacy** ensures drivers remain anonymous.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the V2X PKI (Public Key Infrastructure) hierarchy (Root CA, Enrollment CA, Pseudonym CA).
2.  **Implement** Digital Signatures using ECDSA (Elliptic Curve Digital Signature Algorithm).
3.  **Simulate** Certificate Rotation (changing IDs to prevent tracking).
4.  **Analyze** the "Butterfly Effect" of a Sybil Attack (Fake cars).
5.  **Verify** a signed message to ensure integrity and authenticity.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Cryptography:** Public/Private Keys, Hashing (SHA-256).
-   **Day 86:** BSM Structure.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `cryptography` (OpenSSL wrapper).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Trust Model

How do I trust a message from a stranger?
-   **Digital Signature:** The sender signs the message with their Private Key.
-   **Verification:** The receiver verifies it with the Sender's Public Key.
-   **Certificate:** A trusted authority (CA) signs the Sender's Public Key, saying "This key belongs to a valid Honda Civic".

### 🔹 Part 2: SCMS (Security Credential Management System)

The V2X PKI is complex.
1.  **Root CA:** The ultimate trust anchor.
2.  **Enrollment CA (ECA):** Gives the car a long-term "Enrollment Certificate" (like a Passport).
3.  **Pseudonym CA (PCA):** Gives the car short-term "Pseudonym Certificates" (like Tickets).
    -   A car holds ~3000 tickets (20 per week).
    -   It swaps tickets every 5 minutes.

### 🔹 Part 3: Privacy (Anti-Tracking)

If I always sign with Key A, you can track Key A from home to work.
**Solution:**
-   **Pseudonymity:** I sign with Key A (08:00-08:05). Then Key B (08:05-08:10).
-   **Unlinkability:** The PCA ensures Key A and Key B cannot be linked to the same car by an observer.

---

## 💻 Implementation: Secure V2X Messaging

**Scenario:**
-   **Car A:** Generates a key pair, signs a BSM.
-   **Car B:** Verifies the signature.
-   **Hacker:** Tries to modify the BSM or sign with a fake key.

### 🛠️ Setup
Create `week13_day90` and `v2x_security.py`.

```bash
mkdir -p ~/ros2_ws/src/week13_day90
cd ~/ros2_ws/src/week13_day90
touch v2x_security.py
```

### 👨‍💻 Code: ECDSA Signing & Verification

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
import json
import time
import binascii

class V2XCertificate:
    def __init__(self, cert_id):
        self.id = cert_id
        # In reality, this would be signed by a CA
        # Here, we just generate a key pair for the cert
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()
        
    def get_public_pem(self):
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

class SecureVehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        # Pool of Pseudonym Certificates
        self.certs = [V2XCertificate(f"{vehicle_id}_cert_{i}") for i in range(5)]
        self.current_cert_idx = 0
        
    def rotate_cert(self):
        self.current_cert_idx = (self.current_cert_idx + 1) % len(self.certs)
        print(f"[{self.vehicle_id}] Rotated to Cert: {self.certs[self.current_cert_idx].id}")
        
    def sign_message(self, message_dict):
        # 1. Serialize Message (Canonical Form)
        msg_bytes = json.dumps(message_dict, sort_keys=True).encode('utf-8')
        
        # 2. Sign with Current Private Key
        current_cert = self.certs[self.current_cert_idx]
        signature = current_cert.private_key.sign(
            msg_bytes,
            ec.ECDSA(hashes.SHA256())
        )
        
        # 3. Attach Signature and Public Key (Certificate)
        # In V2X, we attach the Certificate Digest, not the full cert (to save bandwidth)
        # But for this demo, we attach the Public Key so receiver can verify
        signed_packet = {
            'payload': message_dict,
            'signature': binascii.hexlify(signature).decode('utf-8'),
            'cert_id': current_cert.id,
            'public_key': current_cert.get_public_pem().decode('utf-8')
        }
        return signed_packet

class Verifier:
    def verify(self, signed_packet):
        try:
            # 1. Extract Data
            payload = signed_packet['payload']
            signature = binascii.unhexlify(signed_packet['signature'])
            pub_key_pem = signed_packet['public_key'].encode('utf-8')
            
            # 2. Reconstruct Message Bytes
            msg_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
            
            # 3. Load Public Key
            public_key = serialization.load_pem_public_key(pub_key_pem)
            
            # 4. Verify
            public_key.verify(signature, msg_bytes, ec.ECDSA(hashes.SHA256()))
            
            print(f"✅ Verification SUCCESS. Msg from {signed_packet['cert_id']}: {payload}")
            return True
            
        except Exception as e:
            print(f"❌ Verification FAILED: {e}")
            return False

def main():
    car = SecureVehicle("Honda_Civic")
    verifier = Verifier()
    
    # --- Scenario 1: Valid Message ---
    print("\n--- Scenario 1: Valid Message ---")
    msg = {'type': 'BSM', 'speed': 30.0, 'lat': 37.0, 'lon': -122.0}
    packet = car.sign_message(msg)
    verifier.verify(packet)
    
    # --- Scenario 2: Tampered Message (Man-in-the-Middle) ---
    print("\n--- Scenario 2: Tampered Message ---")
    packet = car.sign_message(msg)
    
    # Hacker modifies speed!
    packet['payload']['speed'] = 100.0 
    
    verifier.verify(packet)
    
    # --- Scenario 3: Certificate Rotation ---
    print("\n--- Scenario 3: Privacy (Rotation) ---")
    packet1 = car.sign_message(msg)
    print(f"Msg 1 signed by: {packet1['cert_id']}")
    
    car.rotate_cert()
    
    packet2 = car.sign_message(msg)
    print(f"Msg 2 signed by: {packet2['cert_id']}")
    
    # Verifier accepts both (assuming both certs are valid in the CRL)
    verifier.verify(packet1)
    verifier.verify(packet2)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Replay Attack

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** Tampered messages fail verification.
3.  **Experiment:**
    -   Capture a valid packet: `saved_packet = packet`.
    -   Wait 10 seconds.
    -   Re-send `saved_packet`.
    -   **Result:** The Verifier says "SUCCESS".
    -   **Problem:** This is a **Replay Attack**. A hacker recorded a "Braking" message and played it back later to cause a phantom jam.
    -   **Fix:** Check the `timestamp` in the payload. If `current_time - msg_time > 2s`, reject it!

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Performance
**Symptom:** Verification takes 5ms. 100 cars x 10Hz = 1000 verifications/sec. CPU overload.
**Cause:** ECDSA math is heavy.
**Solution:**
    -   **Batch Verification:** Verify multiple signatures at once (math trick).
    -   **Hardware Acceleration:** Use V2X HSM.
    -   **Optimistic Verification:** Only verify messages from *relevant* cars (close and approaching).

#### 2. CRL (Certificate Revocation List)
**Symptom:** A hacked car keeps sending bad data.
**Cause:** We verified the signature, but didn't check if the cert was revoked.
**Solution:** The car must download the CRL (blacklist) daily and check every cert against it.

---

## ⚡ Optimization & Best Practices

### 1. Butterfly Curves
V2X uses **ECQV (Elliptic Curve Qu-Vanstone)** or **ECDSA secp256r1**.
-   These curves are optimized for small key sizes (256 bits) with high security (equivalent to RSA 3072 bits).
-   Essential for keeping packet size small.

### 2. Misbehavior Detection
Cryptography only proves *who* sent it, not *if it's true*.
-   If a valid car sends "Speed 100" but GPS says it moved 5 meters in 1s (Speed 5), it's lying.
-   **Local Misbehavior Detection:** Cross-check V2X data with onboard sensors (Radar/Camera).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between Encryption and Signing?
    *   **A:** **Encryption** hides the data (Privacy). **Signing** proves the origin and integrity (Authenticity). V2X BSMs are **Signed** but NOT Encrypted (Safety messages must be readable by everyone).
2.  **Q:** Why do we need Pseudonym Certificates?
    *   **A:** To prevent tracking. If we used one permanent certificate, the car's path could be logged forever.
3.  **Q:** What is a Sybil Attack?
    *   **A:** One hacker pretending to be 100 cars (using 100 stolen certs) to create a fake traffic jam.

### Challenge Task
**Task:** Timestamp Check.
1.  Add `ts` (timestamp) to the message.
2.  Modify `Verifier` to check: `if abs(time.time() - msg['ts']) > 2.0: return False`.
3.  Try the Replay Attack again. It should fail.

---

## 📚 Further Reading & References
-   [IEEE 1609.2 (Security Services for WAVE)](https://standards.ieee.org/standard/1609_2-2016.html)
-   [SCMS Proof of Concept](https://www.its.dot.gov/pilots/pdf/SCMS_POC_System_Design.pdf)

---

**Day 90 Complete** | Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication
