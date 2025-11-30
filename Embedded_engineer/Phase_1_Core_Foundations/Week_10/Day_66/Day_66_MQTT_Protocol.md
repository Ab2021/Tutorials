# Day 66: MQTT Protocol Implementation
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
1.  **Explain** the Publish-Subscribe architecture of MQTT.
2.  **Differentiate** between QoS Levels 0, 1, and 2.
3.  **Implement** an MQTT Client using the LwIP MQTT Application.
4.  **Publish** sensor data to a public broker.
5.  **Subscribe** to a topic and control an LED remotely.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board with Ethernet.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   MQTT Explorer (PC Tool).
*   **Prior Knowledge:**
    *   Day 65 (LwIP Sockets)
*   **Datasheets:**
    *   [MQTT 3.1.1 Specification](http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Pub-Sub Model
*   **Client:** The STM32.
*   **Broker:** The Server (e.g., Mosquitto, AWS IoT).
*   **Topic:** A path string (e.g., `home/livingroom/temp`).
*   **Publish:** Client sends data to a Topic.
*   **Subscribe:** Client asks Broker to send any data published to a Topic.
*   **Decoupling:** The Publisher doesn't know who the Subscriber is.

### 🔹 Part 2: QoS (Quality of Service)
*   **QoS 0 (At most once):** Fire and Forget. No ACK. Fastest.
*   **QoS 1 (At least once):** Sender waits for PUBACK. Retries if needed. Duplicates possible.
*   **QoS 2 (Exactly once):** 4-step handshake. Slowest, most reliable.

---

## 💻 Implementation: LwIP MQTT Client

> **Instruction:** We will use the `lwip/apps/mqtt.h` API. It is callback-based (Raw API style) but works with RTOS.

### 👨‍💻 Code Implementation

#### Step 1: Init & Connect
```c
#include "lwip/apps/mqtt.h"

mqtt_client_t *client;

void mqtt_connection_cb(mqtt_client_t *client, void *arg, mqtt_connection_status_t status) {
    if (status == MQTT_CONNECT_ACCEPTED) {
        printf("MQTT Connected!\n");
        
        // Subscribe
        mqtt_sub_unsub(client, "stm32/led", 0, mqtt_request_cb, NULL, 1);
    } else {
        printf("MQTT Connect Failed: %d\n", status);
    }
}

void MQTT_Start(void) {
    client = mqtt_client_new();
    
    struct mqtt_connect_client_info_t ci;
    memset(&ci, 0, sizeof(ci));
    ci.client_id = "stm32_device_001";
    ci.keep_alive = 60;
    
    ip_addr_t broker_ip;
    IP4_ADDR(&broker_ip, 192, 168, 1, 100); // Local Mosquitto
    
    mqtt_client_connect(client, &broker_ip, 1883, mqtt_connection_cb, NULL, &ci);
}
```

#### Step 2: Publish
```c
void mqtt_pub_request_cb(void *arg, err_t result) {
    if (result != ERR_OK) printf("Pub Failed\n");
}

void MQTT_Publish_Temp(float temp) {
    char payload[32];
    sprintf(payload, "{\"temp\": %.2f}", temp);
    
    mqtt_publish(client, "stm32/temp", payload, strlen(payload), 0, 0, mqtt_pub_request_cb, NULL);
}
```

#### Step 3: Subscribe Callback
```c
// Called when data arrives
void mqtt_incoming_data_cb(void *arg, const u8_t *data, u16_t len, u8_t flags) {
    char buf[64];
    if (len > 63) len = 63;
    memcpy(buf, data, len);
    buf[len] = 0;
    
    printf("Msg: %s\n", buf);
    
    if (strcmp(buf, "ON") == 0) {
        LED_On();
    } else if (strcmp(buf, "OFF") == 0) {
        LED_Off();
    }
}

// Called when topic matches
void mqtt_incoming_publish_cb(void *arg, const char *topic, u32_t tot_len) {
    printf("Topic: %s\n", topic);
    // Setup data callback
    mqtt_set_inpub_callback(client, mqtt_incoming_publish_cb, mqtt_incoming_data_cb, NULL);
}
```

---

## 🔬 Lab Exercise: Lab 66.1 - Remote Control

### 1. Lab Objectives
- Use MQTT Explorer to control the STM32 LED.
- View STM32 temperature data on PC.

### 2. Step-by-Step Guide

#### Phase A: Setup Broker
1.  Install Mosquitto on PC (or use `test.mosquitto.org`).
2.  Run `mosquitto -v`.

#### Phase B: STM32 Code
1.  Flash code.
2.  Wait for "MQTT Connected!".

#### Phase C: Test
1.  Open MQTT Explorer. Connect to Broker.
2.  Subscribe to `stm32/#`.
3.  **Observation:** You should see `stm32/temp` updating.
4.  Publish to `stm32/led` with payload "ON".
5.  **Observation:** STM32 LED turns ON.

### 3. Verification
If connection fails, check Firewall (Port 1883).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Last Will & Testament (LWT)
- **Goal:** Detect unexpected disconnection.
- **Task:**
    1.  Set `ci.will_topic = "stm32/status"`.
    2.  Set `ci.will_msg = "OFFLINE"`.
    3.  Set `ci.will_qos = 1`.
    4.  Connect. Publish "ONLINE" to `stm32/status`.
    5.  Pull the plug (Ethernet).
    6.  **Observation:** Broker publishes "OFFLINE" to the topic after Keep Alive timeout.

### Lab 3: Retained Messages
- **Goal:** New subscribers get last known state.
- **Task:**
    1.  Publish Temp with `retained = 1`.
    2.  Disconnect MQTT Explorer. Reconnect.
    3.  **Observation:** You immediately receive the last Temp value without waiting for the next update.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Connection Refused (Code -4)
*   **Cause:** Broker not running, or Authentication failed (Username/Password).
*   **Solution:** Check Broker logs.

#### 2. Keep Alive Timeout
*   **Cause:** LwIP thread blocked or not calling `mqtt_cyclic_timer` (if using Raw). In RTOS/Socket mode, the thread handles it.
*   **Solution:** Ensure network is stable.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Topic Naming:** Use hierarchy `Region/DeviceType/ID/Metric`. E.g., `US/Sensor/001/Temp`.
- **JSON:** Use a library (cJSON) to parse/build payloads. Don't use `sprintf` manually for complex data.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the standard MQTT port?
    *   **A:** 1883 (Unencrypted), 8883 (Encrypted/TLS).
2.  **Q:** Does MQTT run over UDP?
    *   **A:** No, standard MQTT uses TCP. MQTT-SN (Sensor Networks) uses UDP.

### Challenge Task
> **Task:** Implement "RPC over MQTT". Publish to `stm32/cmd` with payload `{"id": 123, "method": "get_uptime"}`. STM32 replies to `stm32/resp` with `{"id": 123, "result": 5000}`.

---

## 📚 Further Reading & References
- [HiveMQ MQTT Essentials Blog](https://www.hivemq.com/mqtt-essentials/)

---
