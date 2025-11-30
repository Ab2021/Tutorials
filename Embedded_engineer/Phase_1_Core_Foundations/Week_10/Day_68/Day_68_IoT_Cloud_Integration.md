# Day 68: IoT Cloud Integration (AWS/Azure basics)
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
1.  **Explain** the concept of "Device Shadow" (AWS) or "Device Twin" (Azure).
2.  **Generate** and **Provision** certificates for a cloud connection.
3.  **Implement** the logic to sync local state with the Cloud Shadow.
4.  **Publish** telemetry data to a specific cloud topic structure.
5.  **Handle** cloud commands via Delta topics.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   AWS IoT Core Account (Free Tier) or Azure IoT Hub.
*   **Prior Knowledge:**
    *   Day 66 (MQTT)
    *   Day 67 (TLS)
*   **Datasheets:**
    *   [AWS IoT Device Shadow Service](https://docs.aws.amazon.com/iot/latest/developerguide/iot-device-shadows.html)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Shadow" Concept
IoT devices are unreliable. They sleep. They disconnect.
*   **Problem:** If I send "Turn On" while the device is offline, the command is lost.
*   **Solution:** A JSON document stored in the Cloud (The Shadow).
    *   **Desired State:** What the user wants (e.g., `{"light": "ON"}`).
    *   **Reported State:** What the device last said (e.g., `{"light": "OFF"}`).
*   **Sync:** When the device wakes up, it asks for the Delta (Difference). It sees Desired="ON", turns the light on, and updates Reported="ON".

### 🔹 Part 2: Topics Structure (AWS Example)
*   `$aws/things/MyDevice/shadow/update`: Publish here to update state.
*   `$aws/things/MyDevice/shadow/update/delta`: Subscribe here to receive commands.
*   `$aws/things/MyDevice/shadow/get`: Publish here to request current state.

---

## 💻 Implementation: AWS IoT Shadow Sync

> **Instruction:** We will implement the Shadow logic using our MQTT Client.

### 👨‍💻 Code Implementation

#### Step 1: JSON Construction
```c
#include "cJSON.h"

void Send_Reported_State(int led_state) {
    // Build JSON: {"state": {"reported": {"led": "ON"}}}
    cJSON *root = cJSON_CreateObject();
    cJSON *state = cJSON_CreateObject();
    cJSON *reported = cJSON_CreateObject();
    
    cJSON_AddStringToObject(reported, "led", led_state ? "ON" : "OFF");
    cJSON_AddItemToObject(state, "reported", reported);
    cJSON_AddItemToObject(root, "state", state);
    
    char *payload = cJSON_PrintUnformatted(root);
    
    // Publish
    mqtt_publish(client, "$aws/things/MyDevice/shadow/update", payload, ...);
    
    cJSON_Delete(root);
    free(payload);
}
```

#### Step 2: Delta Callback
```c
void Handle_Delta(char *json_str) {
    // Received: {"state": {"led": "ON"}, "version": 123}
    cJSON *root = cJSON_Parse(json_str);
    cJSON *state = cJSON_GetObjectItem(root, "state");
    cJSON *led = cJSON_GetObjectItem(state, "led");
    
    if (led) {
        if (strcmp(led->valuestring, "ON") == 0) {
            LED_On();
            Send_Reported_State(1); // Confirm
        } else {
            LED_Off();
            Send_Reported_State(0); // Confirm
        }
    }
    cJSON_Delete(root);
}
```

#### Step 3: Main Logic
```c
void AWS_Task(void *p) {
    // 1. Connect TLS (Port 8883)
    // 2. Connect MQTT
    // 3. Subscribe to Delta
    mqtt_subscribe(client, "$aws/things/MyDevice/shadow/update/delta", 1);
    
    // 4. Request current state (in case we missed something while offline)
    mqtt_publish(client, "$aws/things/MyDevice/shadow/get", "", 0, 0, 0, NULL, NULL);
    
    while(1) {
        // Send Telemetry
        Send_Telemetry();
        vTaskDelay(5000);
    }
}
```

---

## 🔬 Lab Exercise: Lab 68.1 - The Digital Twin

### 1. Lab Objectives
- Create a "Thing" in AWS IoT Core.
- Download Certs.
- Control the STM32 via the AWS Console.

### 2. Step-by-Step Guide

#### Phase A: AWS Console
1.  Go to IoT Core -> Manage -> Things -> Create Thing.
2.  Name: "STM32_F4".
3.  Auto-generate certificates. Download `xxx-certificate.pem.crt` and `xxx-private.pem.key`.
4.  Attach Policy (Allow `iot:*`).

#### Phase B: STM32
1.  Convert PEM to C Header (const char array).
2.  Load into mbedTLS.
3.  Flash and Run.

#### Phase C: Test
1.  In AWS Console -> Device Shadow.
2.  Edit "Desired State": `{"led": "ON"}`.
3.  **Observation:** STM32 LED turns ON. Shadow "Reported State" updates to "ON".

### 3. Verification
If "Reported" doesn't update, check if the device has permission to Publish to the update topic.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Azure IoT Hub (SAS Token)
- **Goal:** Connect to Azure.
- **Task:**
    1.  Azure uses Username/Password instead of Mutual TLS (sometimes).
    2.  Username: `HostName=MyHub.azure-devices.net;DeviceId=MyDevice`.
    3.  Password: Shared Access Signature (SAS) Token (HMAC-SHA256 of URI + Expiry).
    4.  Generate SAS Token on PC or STM32 (needs Base64 & HMAC).

### Lab 3: Rules Engine
- **Goal:** Database Storage.
- **Task:**
    1.  Create a Rule in AWS: `SELECT * FROM 'stm32/data'`.
    2.  Action: Insert into DynamoDB.
    3.  Publish data. Check Database.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. TLS Error -0x2700 (X509 - Cert Verify Failed)
*   **Cause:** STM32 doesn't trust AWS Root CA.
*   **Solution:** Download "Amazon Root CA 1" and include it in `cacert`.

#### 2. Disconnects immediately
*   **Cause:** Policy doesn't allow Client ID.
*   **Solution:** Check AWS Policy. Resource: `arn:aws:iot:us-east-1:xxx:client/stm32_device_001`.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **JIT Provisioning:** Don't hardcode certs. Use a "Claim Certificate" to connect first, then request a unique certificate from the cloud (Fleet Provisioning).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between Telemetry and State?
    *   **A:** Telemetry is a stream of data (Temp=25, Temp=26). State is a persistent property (Color=Red). Shadows are for State.
2.  **Q:** Why use JSON?
    *   **A:** Human readable, standard, flexible. But verbose. Use CBOR for binary efficiency if supported.

### Challenge Task
> **Task:** Implement "Offline Buffering". If MQTT disconnects, store telemetry in a Ring Buffer (RAM or Flash). When reconnected, burst publish the buffered data with original timestamps.

---

## 📚 Further Reading & References
- [AWS IoT Embedded C SDK](https://github.com/aws/aws-iot-device-sdk-embedded-C)

---
