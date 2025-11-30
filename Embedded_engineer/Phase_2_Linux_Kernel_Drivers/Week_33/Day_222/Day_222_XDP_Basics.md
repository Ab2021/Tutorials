# Day 222: XDP (eXpress Data Path) Basics
## Phase 2: Linux Kernel & Device Drivers | Week 33: Advanced Network Drivers

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
1.  **Explain** where XDP hooks into the driver (Before `skb` allocation).
2.  **Understand** the XDP actions (`XDP_DROP`, `XDP_PASS`, `XDP_TX`, `XDP_REDIRECT`).
3.  **Prepare** a driver to support XDP (`ndo_bpf`).
4.  **Visualize** the performance benefits (DDOS mitigation, Load Balancing).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Clang/LLVM (to compile BPF).
    *   `bpftool`.
*   **Prior Knowledge:**
    *   Day 218 (RX Ring).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Skb" Overhead
*   Allocating `struct sk_buff` and parsing headers takes time (~100s of nanoseconds).
*   For 10GbE+ line rate, this is too slow for simple tasks like dropping packets (Firewall).
*   **XDP:** Runs a BPF program on the raw DMA buffer *before* the OS touches it.

### 🔹 Part 2: The Hook Point
*   **Driver RX Loop:**
    1.  Sync DMA.
    2.  **RUN XDP PROGRAM.**
    3.  If `XDP_DROP`: Recycle buffer. Done.
    4.  If `XDP_PASS`: Alloc `skb`, pass to stack.
    5.  If `XDP_TX`: Swap MACs, send back out same interface (Hairpin).

---

## 💻 Implementation: Driver Support for XDP

> **Instruction:** Modify the RX loop to call `bpf_prog_run_xdp`.

### 👨‍💻 Code Implementation

```c
#include <linux/bpf.h>
#include <linux/filter.h>

// In RX Loop
struct bpf_prog *xdp_prog = READ_ONCE(priv->xdp_prog);
struct xdp_buff xdp;
u32 act = XDP_PASS;

if (xdp_prog) {
    // Setup xdp_buff
    xdp.data = packet_start;
    xdp.data_end = packet_start + length;
    xdp.data_hard_start = buffer_start;
    xdp.rxq = &priv->xdp_rxq;
    
    // Run BPF
    act = bpf_prog_run_xdp(xdp_prog, &xdp);
    
    switch (act) {
        case XDP_DROP:
            recycle_buffer(priv, buf);
            goto next_packet;
        case XDP_TX:
            // Transmit immediately
            my_xdp_xmit(priv, &xdp);
            goto next_packet;
        case XDP_REDIRECT:
            xdp_do_redirect(netdev, &xdp, xdp_prog);
            goto next_packet;
        case XDP_PASS:
        default:
            break; // Fall through to skb alloc
    }
}

// ... Alloc skb ...
```

---

## 🔬 Lab Exercise: Lab 222.1 - Loading a Dummy XDP

### 1. Lab Objectives
- Compile a BPF program that drops everything.
- Attach it to `lo` (Loopback) or a real interface.

### 2. Step-by-Step Guide
1.  **Code (drop.c):**
    ```c
    #include <linux/bpf.h>
    #include <bpf/bpf_helpers.h>

    SEC("xdp")
    int xdp_drop_all(struct xdp_md *ctx) {
        return XDP_DROP;
    }
    char _license[] SEC("license") = "GPL";
    ```
2.  **Compile:**
    ```bash
    clang -O2 -target bpf -c drop.c -o drop.o
    ```
3.  **Load:**
    ```bash
    ip link set dev eth0 xdp obj drop.o sec xdp
    ```
4.  **Test:** Ping eth0. It should fail.
5.  **Unload:**
    ```bash
    ip link set dev eth0 xdp off
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: XDP_TX (Packet Generator)
- **Goal:** Modify packet and send back.
- **Task:**
    *   Swap Source/Dest MAC.
    *   Return `XDP_TX`.
    *   Result: Instant echo server without CPU overhead.

### Lab 3: XDP Metadata
- **Goal:** Pass info to skb.
- **Task:**
    *   XDP program writes metadata before `data`.
    *   Driver reads it and populates `skb` fields (e.g., hardware timestamp or hash).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Operation not supported"
*   **Cause:** Driver does not implement `ndo_bpf`.
*   **Fix:** Use `xdpgeneric` (Generic XDP).
    *   `ip link set dev eth0 xdpgeneric ...`
    *   Runs in software (after skb alloc), slower but works on all drivers.

#### 2. Headroom issues
*   **Cause:** XDP requires `XDP_PACKET_HEADROOM` (256 bytes) before the packet data for header manipulation (encapsulation).
*   **Fix:** Ensure DMA buffer starts at offset 256.

---

## ⚡ Optimization & Best Practices

### JIT Compiler
*   Ensure `bpf_jit_enable` is 1.
*   XDP without JIT is slow (interpreter).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is XDP faster than Netfilter?
    *   **A:** Netfilter runs after `skb` allocation and parsing. XDP runs on raw bytes.
2.  **Q:** Can XDP access packet payload?
    *   **A:** Yes, it can read/write any part of the packet.

### Challenge Task
> **Task:** "The Firewall".
> *   Write an XDP program that drops packets from a specific IP (parse Ethernet -> IP header).
> *   Attach to your driver.
> *   Benchmark `hping3` flood. CPU usage should remain low.

---

## 📚 Further Reading & References
- [XDP Tutorial](https://github.com/xdp-project/xdp-tutorial)
- [Cilium BPF Documentation](https://docs.cilium.io/en/stable/bpf/)

---
