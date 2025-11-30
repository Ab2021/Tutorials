# Day 46: USB CDC Class (Virtual COM Port)
## Phase 1: Core Embedded Engineering Foundations | Week 7: Advanced Peripherals

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
1.  **Understand** the CDC-ACM (Abstract Control Model) subclass.
2.  **Implement** the class-specific requests (`SET_LINE_CODING`, `GET_LINE_CODING`).
3.  **Transmit** and **Receive** data using Bulk Endpoints.
4.  **Integrate** `printf` to output to the USB Virtual COM Port.
5.  **Achieve** high-speed data transfer (> 500 KB/s) compared to UART.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Micro-USB Cable
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   Terminal (PuTTY/TeraTerm)
*   **Prior Knowledge:**
    *   Day 45 (USB Basics)
*   **Datasheets:**
    *   [USB CDC Specification](https://www.usb.org/sites/default/files/CDC1.2_WMC1.1_012011.zip)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: CDC-ACM Overview
The Communication Device Class (CDC) allows the USB device to look like a legacy Serial Port (COMx on Windows, /dev/ttyACM0 on Linux).
*   **Interfaces:**
    1.  **Communication Interface:** Uses Interrupt Endpoint. Sends notifications (e.g., Ring Detect, Serial State).
    2.  **Data Interface:** Uses Bulk IN and Bulk OUT Endpoints. Carries the actual stream.

### 🔹 Part 2: Line Coding
Even though USB doesn't have "Baud Rate" (it's always 12 Mbps), the OS driver expects to set it.
*   **SET_LINE_CODING (0x20):** Host sends Baud, Stop Bits, Parity, Data Bits. Device stores it.
*   **GET_LINE_CODING (0x21):** Host asks for current settings. Device replies.
*   **Control_Line_State (0x22):** Host sets DTR/RTS signals.

---

## 💻 Implementation: CDC Class Logic

> **Instruction:** We assume the USB Stack (Driver) handles the standard requests. We implement the Class Callbacks.

### 👨‍💻 Code Implementation

#### Step 1: Line Coding Structure
```c
typedef struct {
    uint32_t bitrate;
    uint8_t  format; // Stop bits
    uint8_t  paritytype;
    uint8_t  datatype; // Data bits (8)
} USBD_CDC_LineCodingTypeDef;

USBD_CDC_LineCodingTypeDef linecoding = {
    115200, 0x00, 0x00, 0x08
};
```

#### Step 2: Handling Setup Packets
```c
uint8_t USBD_CDC_Setup(USBD_SetupReqTypedef *req) {
    switch (req->bRequest) {
        case 0x20: // SET_LINE_CODING
            // Receive 7 bytes of data from Host (EP0 OUT)
            USBD_CtlPrepareRx(req->wLength);
            return USBD_OK;
            
        case 0x21: // GET_LINE_CODING
            // Send 7 bytes to Host (EP0 IN)
            USBD_CtlSendData((uint8_t*)&linecoding, 7);
            return USBD_OK;
            
        case 0x22: // SET_CONTROL_LINE_STATE
            // DTR/RTS handling (Optional)
            return USBD_OK;
    }
    return USBD_FAIL;
}
```

#### Step 3: Data Transmission (Bulk IN)
```c
uint8_t CDC_Transmit_FS(uint8_t* Buf, uint16_t Len) {
    uint8_t result = USBD_OK;
    
    // Check if USB is configured
    if (hUsbDeviceFS.dev_state != USBD_STATE_CONFIGURED) {
        return USBD_FAIL;
    }
    
    // Check if TX FIFO is ready
    // ...
    
    // Send Data via EP1 IN
    USBD_LL_Transmit(&hUsbDeviceFS, CDC_IN_EP, Buf, Len);
    
    return result;
}
```

#### Step 4: Data Reception (Bulk OUT)
```c
// Called by Stack when data arrives on EP1 OUT
uint8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len) {
    // Process Buf (e.g., put into Ring Buffer)
    RingBuffer_Write(Buf, *Len);
    
    // Prepare for next packet
    USBD_LL_PrepareReceive(&hUsbDeviceFS, CDC_OUT_EP, UserRxBufferFS, CDC_DATA_FS_MAX_PACKET_SIZE);
    
    return USBD_OK;
}
```

---

## 🔬 Lab Exercise: Lab 46.1 - USB Echo

### 1. Lab Objectives
- Type characters in PuTTY.
- See them echoed back by the STM32.

### 2. Step-by-Step Guide

#### Phase A: Main Loop
```c
int main(void) {
    USB_Init();
    
    while(1) {
        if (RingBuffer_Available()) {
            char c = RingBuffer_Read();
            // Echo back
            CDC_Transmit_FS(&c, 1);
        }
    }
}
```

#### Phase B: Testing
1.  Connect USB.
2.  Open Device Manager. Find "STMicroelectronics Virtual COM Port (COMx)".
3.  Open PuTTY on COMx. Baud rate doesn't matter (USB ignores it).
4.  Type "Hello". See "Hello".

### 3. Verification
If it works, you have a 12 Mbps serial link!

---

## 🧪 Additional / Advanced Labs

### Lab 2: High Speed Throughput
- **Goal:** Measure speed.
- **Task:**
    1.  Host runs a Python script to read data.
    2.  Device sends 1 KB packets in a loop.
    3.  Calculate MB/s.
    4.  Expected: ~800 KB/s to 1 MB/s (Full Speed limit is 1.5 MB/s theoretical, overhead reduces it).

### Lab 3: printf Retargeting
- **Goal:** Use `printf` over USB.
- **Task:**
    1.  Override `_write`.
    2.  Call `CDC_Transmit_FS`.
    3.  **Warning:** `CDC_Transmit_FS` is non-blocking (usually). If you call `printf` too fast, you might overwrite the buffer before it's sent. Implement a blocking wait or a large TX buffer.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Windows Driver Missing
*   **Cause:** Windows 7/8 need a driver (.inf file). Windows 10/11 usually loads `usbser.sys` automatically if `bDeviceClass = 0x02`.
*   **Solution:** Install ST VCP Driver or use Zadig.

#### 2. Device Hangs on Transmit
*   **Cause:** Trying to transmit while previous transfer is not complete.
*   **Solution:** Check `TxState` before calling Transmit. Wait until `TxState == 0`.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Double Buffering:** Use double buffering for Bulk Endpoints to maximize throughput. While one packet is being sent over USB, the CPU fills the next one.

### Code Quality
- **ZLP (Zero Length Packet):** If a transfer is exactly a multiple of MaxPacketSize (64), you must send a ZLP to indicate end of transfer. Most stacks handle this, but verify.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Does changing the Baud Rate in PuTTY change the USB speed?
    *   **A:** No. USB Full Speed is fixed at 12 Mbps. The baud rate setting is just a message sent to the device (which it can ignore or use to configure a real UART bridge).
2.  **Q:** Why do we need a separate Interrupt Endpoint?
    *   **A:** For notifications like "Serial State" (DCD, DSR, Break). Data goes over Bulk.

### Challenge Task
> **Task:** Implement a "USB-to-UART Bridge". Data received on USB CDC is sent out via UART1. Data received on UART1 is sent to USB CDC. (Like an FTDI chip).

---

## 📚 Further Reading & References
- [STM32 USB CDC Host/Device Lib](https://www.st.com/en/embedded-software/stsw-stm32121.html)

---
