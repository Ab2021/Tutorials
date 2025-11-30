# Day 223: Writing an XDP Program
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
1.  **Parse** Ethernet, IP, and UDP headers in BPF.
2.  **Use** BPF Maps (Hash, Array) to store state.
3.  **Pass** the BPF Verifier (Boundary checks).
4.  **Count** packets per protocol.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   `libbpf-dev`.
*   **Prior Knowledge:**
    *   Day 222 (XDP Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Verifier
*   **Safety:** The kernel must ensure BPF programs don't crash or hang.
*   **Checks:**
    *   No infinite loops (bounded loops allowed in newer kernels).
    *   No out-of-bounds memory access.
    *   Every pointer access must be checked against `data_end`.

### 🔹 Part 2: BPF Maps
*   **Communication:** How BPF talks to Userspace.
*   **Types:**
    *   `BPF_MAP_TYPE_ARRAY`: Fast, fixed size.
    *   `BPF_MAP_TYPE_HASH`: Key-Value store.
    *   `BPF_MAP_TYPE_PERCPU_ARRAY`: Scalable counters.

---

## 💻 Implementation: Protocol Counter

> **Instruction:** Count packets based on IP Protocol (TCP/UDP/ICMP).

### 👨‍💻 Code Implementation

```c
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>

// Define Map
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 256); // Index = Protocol ID
    __type(key, __u32);
    __type(value, __u64);
} proto_stats SEC(".maps");

SEC("xdp")
int xdp_prog_count(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    struct ethhdr *eth = data;
    struct iphdr *ip;
    __u64 *value;
    __u32 key;

    // 1. Check Ethernet Header
    if (data + sizeof(*eth) > data_end)
        return XDP_PASS;

    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;

    // 2. Check IP Header
    ip = data + sizeof(*eth);
    if ((void *)ip + sizeof(*ip) > data_end)
        return XDP_PASS;

    // 3. Lookup Map
    key = ip->protocol;
    value = bpf_map_lookup_elem(&proto_stats, &key);
    if (value) {
        __sync_fetch_and_add(value, 1);
    }

    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
```

---

## 🔬 Lab Exercise: Lab 223.1 - Reading the Map

### 1. Lab Objectives
- Load the program.
- Use `bpftool` to read the counter.

### 2. Step-by-Step Guide
1.  **Compile & Load:** As before.
2.  **Generate Traffic:** Ping (ICMP=1), Netcat (TCP=6, UDP=17).
3.  **Read Map:**
    ```bash
    bpftool map list
    # Find ID of proto_stats
    bpftool map dump id <ID>
    ```
    *   Look for Key 1 (ICMP), Key 6 (TCP).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Port Redirection
- **Goal:** Redirect port 80 to 8080.
- **Task:**
    1.  Parse TCP header.
    2.  If `dest == 80`, set `dest = 8080`.
    3.  Recalculate Checksum (Incremental update).
    4.  `XDP_PASS`.

### Lab 3: Atomic Counters
- **Goal:** Thread safety.
- **Task:**
    *   `__sync_fetch_and_add` is atomic.
    *   For better performance, use `BPF_MAP_TYPE_PERCPU_ARRAY`. No locking needed, just sum up userspace side.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Verifier Rejects Code
*   **Cause:** "invalid access to packet".
*   **Fix:** You forgot `if (ptr + size > data_end) return XDP_DROP;`. You must check *every* header level.

#### 2. Checksum Errors
*   **Cause:** Modifying packet without updating checksum.
*   **Fix:** Use `bpf_csum_diff` helper.

---

## ⚡ Optimization & Best Practices

### Loop Unrolling
*   The verifier hates loops. Use `#pragma unroll` if iterating over fixed size data.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `xdp_md`?
    *   **A:** The metadata context passed to the BPF program. Contains `data`, `data_end`, `ingress_ifindex`, etc.
2.  **Q:** Can I call kernel functions from XDP?
    *   **A:** No. Only BPF Helpers (`bpf_map_lookup`, `bpf_ktime_get_ns`, etc.).

### Challenge Task
> **Task:** "The DDoS Shield".
> *   Create a `BPF_MAP_TYPE_HASH` (Source IP -> Count).
> *   If an IP sends > 100 packets/sec, add to Block List.
> *   Drop packets from Block List.

---

## 📚 Further Reading & References
- [IOVisor BPF Docs](https://github.com/iovisor/bpf-docs)

---
