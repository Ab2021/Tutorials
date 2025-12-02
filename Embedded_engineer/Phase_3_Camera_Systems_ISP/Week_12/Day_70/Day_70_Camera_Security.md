# Day 70: Camera Security (Authentication & Encryption)
## Phase 3: Camera Systems & ISP | Week 12: Advanced Automotive Topics

---

## 🎯 Learning Objectives
1.  **Understand** the Threat Model: Spoofing, Tampering, Eavesdropping.
2.  **Implement** Camera Authentication: Ensuring the camera is genuine (not a cheap clone).
3.  **Secure** the Video Link: HDCP (High-bandwidth Digital Content Protection) on GMSL/FPD-Link.
4.  **Encrypt** stored video: AES-256 for Dashcam footage.
5.  **Explore** Secure Boot and TrustZone for camera drivers.
6.  **Analyze** "Privacy Mode" implementation (Hardware Kill Switch).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera Module with Authenticator Chip (e.g., Maxim DS28E15) or Secure Element.
*   **Software:** OpenSSL, Linux Kernel Crypto API.
*   **Knowledge:** Symmetric vs Asymmetric Encryption (AES vs RSA/ECC).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Secure Cameras?
*   **ADAS Spoofing:** Attacker injects a fake image (e.g., empty road) into the AEB system -> Crash.
*   **Privacy:** Hacker watches the driver via DMS (Driver Monitoring System).
*   **Counterfeit Parts:** Cheap replacement cameras with poor latency/quality cause safety issues.

### 🔹 Part 2: Authentication (Challenge-Response)
*   **Goal:** ECU verifies the Camera is genuine.
*   **Mechanism:**
    1.  Camera has a unique Private Key (burned in factory).
    2.  ECU sends a Random Number (Nonce).
    3.  Camera signs the Nonce with Private Key.
    4.  ECU verifies Signature with Public Key.
*   **Protocol:** MIPI C-PHY/D-PHY does not support this natively. Done via I2C (Sideband) or GMSL/FPD-Link security features.

### 🔹 Part 3: Link Encryption (HDCP)
*   **HDCP (High-bandwidth Digital Content Protection):**
    *   Originally for HDMI (Hollywood movies).
    *   Now used in Automotive SerDes (GMSL2/3) to prevent "Man-in-the-Middle" attacks on the video stream.
    *   Encrypts the pixels on the wire.

---

## 💻 Implementation Examples

### Example 1: Challenge-Response (Pseudo-Code)

Verifying a camera module via I2C.

```c
bool verify_camera_authenticity(int i2c_fd) {
    uint8_t nonce[32];
    get_random_bytes(nonce, 32);
    
    // 1. Send Challenge
    i2c_write(i2c_fd, REG_AUTH_CHALLENGE, nonce, 32);
    
    // 2. Wait for calculation (SHA-256 / ECDSA)
    usleep(50000);
    
    // 3. Read Response (Signature)
    uint8_t signature[64];
    i2c_read(i2c_fd, REG_AUTH_RESPONSE, signature, 64);
    
    // 4. Verify (Using Public Key stored in ECU)
    return crypto_verify_signature(public_key, nonce, signature);
}
```

### Example 2: AES Encryption for Storage

Encrypting dashcam video before saving to SD Card.

```bash
# Using OpenSSL (Command Line)
openssl enc -aes-256-cbc -salt -in video.mp4 -out video.enc -k mypassword

# Using GStreamer (Pipeline)
gst-launch-1.0 \
    v4l2src ! video/x-raw,width=1920,height=1080 ! \
    x264enc ! h264parse ! \
    aesenc key=00112233445566778899aabbccddeeff iv=00000000000000000000000000000000 ! \
    mp4mux ! filesink location=secure_video.mp4
```

### Example 3: Secure Boot (Chain of Trust)

Ensuring the Camera Driver hasn't been tampered with.
1.  **BootROM:** Verifies Bootloader (U-Boot).
2.  **U-Boot:** Verifies Kernel (`zImage.signed`).
3.  **Kernel:** Verifies Modules (`camera.ko.signed`).
4.  **DM-Verity:** Verifies Root Filesystem (Read-Only).

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Simulating an Attack

**Objective:** Man-in-the-Middle (Replay Attack).

**Steps:**
1.  Record the I2C traffic during authentication (Logic Analyzer).
2.  Disconnect the camera.
3.  Connect a microcontroller (Attacker) to the I2C bus.
4.  Replay the recorded "Signature".
5.  **Result:** If the ECU sends a *fixed* challenge, the attack works. If the ECU sends a *random* nonce, the attack fails (Signature mismatch).

### Lab 2: Encrypted Streaming

**Objective:** SRTP (Secure RTP).

**Steps:**
1.  Set up GStreamer RTSP server (Day 52).
2.  Enable SRTP (`srtpenc`).
3.  Capture packets with Wireshark.
4.  **Observation:** The payload is garbage (Encrypted). You cannot view the video without the key.

### Lab 3: Privacy LED

**Objective:** Hardware indicator.

**Steps:**
1.  Wire an LED to the Camera Module's power rail (or a GPIO controlled by the Sensor's VSYNC).
2.  **Goal:** LED *must* light up whenever the sensor is streaming.
3.  **Security:** This should be hardware-controlled, not software-controlled (Software can be hacked to turn off LED while recording).

---

## 🐛 Debugging Security Issues

### Debug 1: Authentication Failure

**Symptom:** Camera works for 5 seconds, then cuts out.

**Cause:**
*   ECU performs periodic auth checks. Check failed.
*   I2C noise corrupted the signature.
*   **Fix:** Check signal integrity. Implement retry logic (3 strikes rule).

### Debug 2: HDCP Link Failure

**Symptom:** No video, or "Snow" screen.

**Cause:**
*   HDCP Keys not provisioned in the SerDes chips.
*   Display does not support HDCP.
*   **Fix:** Verify HDCP Key injection status in SerDes registers.

---

## ⚡ Performance Optimization

### Optimization 1: Hardware Crypto Engine

*   Software AES (CPU) is slow and consumes battery.
*   Use the SoC's **Hardware Crypto Accelerator** (CAAM on NXP, CE on Allwinner).
*   `af_alg` interface in Linux.

### Optimization 2: TrustZone (TEE)

*   Store Private Keys in the **Trusted Execution Environment** (TEE) (e.g., OP-TEE).
*   The Main OS (Android/Linux) never sees the keys. It asks the TEE to "Sign this data".
*   Prevents key theft even if the OS is rooted.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is a "Nonce" and why is it used?** (Number Used Once - prevents Replay Attacks).
2.  **Difference between HDCP and AES?** (HDCP protects the link/wire; AES protects the file/data).
3.  **Why is "Secure Boot" necessary for Camera Safety?** (To prevent malware from modifying the driver/ISP tuning).
4.  **What is a "Physical Unclonable Function" (PUF)?** (Silicon fingerprint used to generate keys).

### Practical Challenges

1.  **Implement "Secure Wipe":** Write a script that overwrites the SD card with zeros/random data, not just deletes the file table.
2.  **Harden the System:** Disable `adb`, UART console, and SSH before shipping the product.

---

## 📚 Further Reading & Resources

### Standards
*   **ISO 21434:** Road Vehicles - Cybersecurity Engineering.
*   **HDCP 2.3 Specification.**

---

## 🎓 Summary

Today we covered:
- ✅ **Threats:** Spoofing, Cloning.
- ✅ **Auth:** Challenge-Response.
- ✅ **Encryption:** AES, HDCP.
- ✅ **Trust:** Secure Boot, TEE.
- ✅ **Privacy:** Hardware indicators.

**Next:** Day 71 - Power Management (Suspend/Resume).

---

**Day 70 Complete** | Phase 3: Camera Systems & ISP | Week 12: Advanced Automotive Topics
