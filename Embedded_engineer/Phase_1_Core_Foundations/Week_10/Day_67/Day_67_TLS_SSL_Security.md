# Day 67: TLS/SSL Security Basics (mbedTLS)
## Phase 1: Core Embedded Engineering Foundations | Week 10: Advanced RTOS & IoT

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the TLS Handshake process (Hello, Certificate, Key Exchange).
2.  **Differentiate** between Root CA, Intermediate CA, and Leaf Certificates.
3.  **Integrate** the mbedTLS library with LwIP Sockets.
4.  **Perform** a secure HTTPS GET request to a public server.
5.  **Optimize** mbedTLS memory usage for embedded systems (RAM constraints).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   mbedTLS Library (Source).
*   **Prior Knowledge:**
    *   Day 65 (Sockets)
    *   Day 54 (Flash - for storing certs)
*   **Datasheets:**
    *   [mbedTLS Knowledge Base](https://tls.mbed.org/kb)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why TLS?
Standard TCP/MQTT is plaintext. Anyone with Wireshark can see your password.
*   **Encryption:** Hides data (AES).
*   **Authentication:** Verifies identity (Certificates).
*   **Integrity:** Ensures data wasn't tampered (HMAC/SHA).

### 🔹 Part 2: The Handshake
1.  **Client Hello:** "I support TLS 1.2, Cipher Suites X, Y, Z."
2.  **Server Hello:** "Let's use TLS 1.2 and Cipher X."
3.  **Certificate:** Server sends its ID card (Signed by a Trusted CA).
4.  **Key Exchange:** Client and Server generate a "Session Key" (Symmetric) using Asymmetric crypto (RSA/ECC).
5.  **Finished:** Switch to Encrypted Mode.

### 🔹 Part 3: mbedTLS
A lightweight SSL library for embedded systems.
*   **Config:** `mbedtls_config.h`. Crucial for reducing ROM/RAM size.
*   **Entropy:** Needs a Random Number Generator (RNG). STM32F4 has a hardware RNG!

---

## 💻 Implementation: Secure Socket (HTTPS)

> **Instruction:** Connect to `www.google.com` (Port 443) and fetch the homepage.

### 👨‍💻 Code Implementation

#### Step 1: RNG Init
```c
#include "mbedtls/entropy.h"
#include "mbedtls/ctr_drbg.h"

mbedtls_entropy_context entropy;
mbedtls_ctr_drbg_context ctr_drbg;

int Hardware_RNG(void *data, unsigned char *output, size_t len) {
    // Use STM32 RNG Peripheral
    for(int i=0; i<len; i+=4) {
        while(!(RNG->SR & 1)); // Wait DRDY
        *(uint32_t*)(output+i) = RNG->DR;
    }
    return 0;
}

void TLS_Init(void) {
    mbedtls_entropy_init(&entropy);
    mbedtls_entropy_add_source(&entropy, Hardware_RNG, NULL, 128, MBEDTLS_ENTROPY_SOURCE_STRONG);
    mbedtls_ctr_drbg_init(&ctr_drbg);
    mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy, "STM32", 5);
}
```

#### Step 2: TLS Context
```c
#include "mbedtls/ssl.h"
#include "mbedtls/net_sockets.h"

mbedtls_ssl_context ssl;
mbedtls_ssl_config conf;
mbedtls_x509_crt cacert;

void Secure_Connect(void) {
    mbedtls_net_context server_fd;
    mbedtls_net_init(&server_fd);
    mbedtls_ssl_init(&ssl);
    mbedtls_ssl_config_init(&conf);
    mbedtls_x509_crt_init(&cacert);
    
    // Load Root CA (GlobalSign or similar for Google)
    // For testing, we might skip verification (NOT SECURE) or load a specific CA.
    // mbedtls_x509_crt_parse(&cacert, (const unsigned char *)mbedtls_test_cas_pem, ...);
    
    // Connect TCP
    mbedtls_net_connect(&server_fd, "www.google.com", "443", MBEDTLS_NET_PROTO_TCP);
    
    // Setup SSL
    mbedtls_ssl_config_defaults(&conf, MBEDTLS_SSL_IS_CLIENT, MBEDTLS_SSL_TRANSPORT_STREAM, MBEDTLS_SSL_PRESET_DEFAULT);
    mbedtls_ssl_conf_authmode(&conf, MBEDTLS_SSL_VERIFY_OPTIONAL); // Allow self-signed for now
    mbedtls_ssl_conf_rng(&conf, mbedtls_ctr_drbg_random, &ctr_drbg);
    
    mbedtls_ssl_setup(&ssl, &conf);
    mbedtls_ssl_set_bio(&ssl, &server_fd, mbedtls_net_send, mbedtls_net_recv);
    
    // Handshake
    printf("Handshaking...\n");
    if (mbedtls_ssl_handshake(&ssl) != 0) {
        printf("Handshake Failed\n");
        return;
    }
    printf("Connected!\n");
    
    // Send GET
    char *msg = "GET / HTTP/1.1\r\nHost: www.google.com\r\n\r\n";
    mbedtls_ssl_write(&ssl, (const unsigned char*)msg, strlen(msg));
    
    // Read Response
    unsigned char buf[1024];
    int len = mbedtls_ssl_read(&ssl, buf, sizeof(buf)-1);
    buf[len] = 0;
    printf("%s\n", buf);
    
    // Cleanup
    mbedtls_ssl_close_notify(&ssl);
    mbedtls_net_free(&server_fd);
    mbedtls_ssl_free(&ssl);
}
```

---

## 🔬 Lab Exercise: Lab 67.1 - MQTTS (Secure MQTT)

### 1. Lab Objectives
- Connect to `test.mosquitto.org` on Port 8883 (TLS).

### 2. Step-by-Step Guide

#### Phase A: Config
1.  Enable `MBEDTLS_SSL_PROTO_TLS1_2`.
2.  Enable `MBEDTLS_CIPHER_MODE_CBC`.
3.  Set `MBEDTLS_SSL_MAX_CONTENT_LEN` to 4096 (Save RAM, standard is 16KB).

#### Phase B: Implementation
1.  Modify the MQTT Socket connect logic.
2.  Instead of passing a raw socket to LwIP MQTT, we need to wrap the LwIP MQTT logic to use mbedTLS read/write functions.
3.  **Easier Path:** Use the `altcp` (Application Layered TCP) API in LwIP which supports TLS natively if mbedTLS is linked.
    *   `altcp_new()` -> `altcp_tls_new()`.
    *   Pass the config.
    *   Use `mqtt_client_connect` with the altcp pcb.

### 3. Verification
Wireshark will show "TLSv1.2 Application Data" instead of "MQTT Publish".

---

## 🧪 Additional / Advanced Labs

### Lab 2: Client Authentication (Mutual TLS)
- **Goal:** Server verifies the Device.
- **Task:**
    1.  Generate Device Key/Cert.
    2.  Load them into mbedTLS (`mbedtls_ssl_conf_own_cert`).
    3.  Connect to a server requiring mTLS (e.g., AWS IoT Core).

### Lab 3: Memory Optimization
- **Goal:** Fit TLS in 64KB RAM.
- **Task:**
    1.  Disable unused Ciphers (Camellia, Aria).
    2.  Reduce Max Fragment Length (MFL) extension usage.
    3.  Use Dynamic Buffer Allocation (`MBEDTLS_MEMORY_BUFFER_ALLOC_C`).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Stack Overflow
*   **Cause:** mbedTLS uses huge stack variables (structs).
*   **Solution:** Increase Task Stack to 4KB or 8KB. Or use `calloc` for context structures.

#### 2. Handshake Fail -0x7xxx
*   **Cause:** Certificate Verification Failed.
*   **Solution:** Check if Root CA is correct. Check System Time! (Certificates have validity dates. If RTC is 1970, cert fails).

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Time Sync:** Always use NTP (Network Time Protocol) to set the RTC before attempting TLS.
- **Cert Storage:** Store Root CAs in Flash (const array) or File System.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between RSA and ECC?
    *   **A:** ECC (Elliptic Curve) provides same security with much smaller keys (256-bit vs 2048-bit RSA). Faster and less RAM. Preferred for IoT.
2.  **Q:** Why do we need a Random Number Generator?
    *   **A:** To generate the "Client Random" and "Pre-Master Secret". If RNG is predictable, the encryption is broken.

### Challenge Task
> **Task:** Implement "Certificate Pinning". Instead of trusting a Root CA, verify that the Server's Public Key hash matches a hardcoded hash. This prevents Man-in-the-Middle even if a CA is compromised.

---

## 📚 Further Reading & References
- [mbedTLS Tutorial](https://tls.mbed.org/kb/how-to/mbedtls-tutorial)

---
