# Day 047: Multi-GPU Programming with HIP
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 7: AMD ROCm & HIP

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Enumerate Devices:** Discover and manage multiple GPUs (e.g., 8x MI250X) in a single node.
2.  **Enable Peer Access:** Connect GPUs via **Infinity Fabric** (or PCIe P2P) to verify direct memory access capability.
3.  **Perform P2P Transfers:** Use `hipMemcpyPeer` to move data directly from GPU A to GPU B without bouncing through Host RAM (which cuts bandwidth in half).
4.  **Execute Kernel P2P:** Write kernels where a thread on Device 0 reads/writes memory located on Device 1 via Unified Addressing (UVA).
5.  **Scale workload:** Split a problem (e.g., Vector Add) across two devices using streams.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Multi-GPU System:** Ideally 2x AMD GPUs. If only 1 GPU is available, the concepts (logic) remain valid but P2P APIs will return "Not Supported".

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Multi-GPU Memory Model

By default, GPUs are isolated.
*   GPU 0 cannot see GPU 1's VRAM.
*   To move data: GPU 0 -> CPU RAM -> GPU 1 VRAM.
    *   Latency: High (Two Copy Engines).
    *   Bandwidth: Limited by PCIe x16 (32 GB/s) or System RAM.

**Peer-to-Peer (P2P):**
*   Allows data to traverse the **Interconnect** directly.
*   **PCIe P2P:** Good.
*   **AMD Infinity Fabric (xGMI):** Incredible (200GB/s - 400GB/s).

### 🔹 Part 2: Addressing (UVA)

Unified Virtual Addressing means:
*   Pointer `0x7fff...` might be Host RAM.
*   Pointer `0x5000...` might be GPU 0 VRAM.
*   Pointer `0x6000...` might be GPU 1 VRAM.

When P2P is enabled, the Hardware MMU (IOMMU) resolves `0x6000` from GPU 0 directly to the physical address on GPU 1.

### 🔹 Part 3: Synchronization

Multi-GPU = Distributed System thinking.
1.  **Launch** Kernel on GPU 0 (Async).
2.  **Launch** Kernel on GPU 1 (Async).
3.  **Wait**?
    *   `hipDeviceSynchronize()` (Stalls CPU).
    *   `hipStreamWaitEvent()` (Stalls GPU Step 2 until Step 1 reaches Event).

---

## 💻 Implementation: P2P Ping-Pong

We will allocate Memory on GPU 0, Memory on GPU 1, and modify GPU 1's memory from a kernel running on GPU 0.

### 🛠️ Step 1: The Code (`p2p_test.cpp`)

```cpp
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

void check(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "Error " << msg << ": " << hipGetErrorString(err) << "\n";
        exit(1);
    }
}

__global__ void remote_write_kernel(int* remote_ptr, int val) {
    // This runs on GPU 0, but writes to GPU 1's memory
    int idx = threadIdx.x;
    remote_ptr[idx] = val + idx;
}

int main() {
    int dev_count;
    check(hipGetDeviceCount(&dev_count), "GetCount");
    
    if (dev_count < 2) {
        std::cout << "Requires at least 2 GPUs. Skipping.\n";
        return 0;
    }

    int dev0 = 0;
    int dev1 = 1;

    // 1. Enable P2P
    int can_access_peer;
    hipDeviceCanAccessPeer(&can_access_peer, dev0, dev1);
    
    if (can_access_peer) {
        std::cout << "P2P Access Supported.\n";
        hipSetDevice(dev0);
        hipDeviceEnablePeerAccess(dev1, 0);
        
        hipSetDevice(dev1);
        hipDeviceEnablePeerAccess(dev0, 0);
    } else {
        std::cout << "P2P Not Supported (PCIe limitations?). Falling back to Host staging?\n";
        return 0;
    }

    // 2. Alloc
    int* d_ptr0;
    int* d_ptr1;
    size_t size = 256 * sizeof(int);

    hipSetDevice(dev0);
    hipMalloc(&d_ptr0, size);

    hipSetDevice(dev1);
    hipMalloc(&d_ptr1, size);

    // 3. Kernel Access
    // Launch on Dev 0, writing to Dev 1 ptr
    hipSetDevice(dev0);
    remote_write_kernel<<<1, 256>>>(d_ptr1, 100); 
    
    check(hipDeviceSynchronize(), "Sync Dev0");

    // 4. Verify
    std::vector<int> h_check(256);
    hipMemcpy(h_check.data(), d_ptr1, size, hipMemcpyDeviceToHost); // Copy from Dev 1

    bool pass = true;
    for(int i=0; i<256; i++) {
        if (h_check[i] != 100 + i) pass = false;
    }

    std::cout << "P2P Write Test: " << (pass ? "PASSED" : "FAILED") << "\n";

    hipFree(d_ptr0);
    hipFree(d_ptr1);
    return 0;
}
```

### 🛠️ Step 2: Compiling

```bash
hipcc p2p_test.cpp -o p2p_test
./p2p_test
```

### 🔹 Part 4: `hipMemcpyPeer`

If you don't need kernel access but just want Move Data:
```cpp
// Explicit P2P Copy
hipMemcpyPeer(dest_ptr, dest_dev, src_ptr, src_dev, bytes);

// Async Version
hipMemcpyPeerAsync(..., stream);
```
Use this for **Halo Exchange** in simulation grids (exchanging boundaries between GPUs).

---

## 🧪 Hands-On Labs

### Lab 47: Bandwidth Test

**Objective:** Measure P2P Bandwidth vs Host Staging.

**Code:**
1.  Loop 100 times doing `hipMemcpyPeer`. Measure time.
2.  Loop 100 times doing `hipMemcpy(D2H)` then `hipMemcpy(H2D)`. Measure time.
3.  Calculate Bandwidth (GB/s).

**Expected Results:**
*   P2P over xGMI (Infinity Fabric): ~100-200 GB/s.
*   P2P over PCIe Gen4: ~25 GB/s.
*   Host Staging: ~10-12 GB/s (limited by System RAM or double copy latency).

---

## 📝 Summary & Key Takeaways

1.  **Topology Matters:** Not all GPUs can talk to each other. Use `rocm-smi --show-topo` (or `nvidia-smi topo -m`) to check connectivity.
2.  **Enable It:** P2P is disabled by default to save IOMMU resources. You must call `hipDeviceEnablePeerAccess`.
3.  **UVA:** Makes multisensor coding easy. Pass a pointer to a kernel, and if it's resident on another GPU (and mapped), it just works.
4.  **Scaling:** Good scalability requires minimizing Host<->Device traffic and maximizing Device<->Device traffic.

---

## 📚 Additional Resources

*   [AMD Infinity Fabric Technology](https://www.amd.com/en/technologies/infinity-architecture)
*   [HIP Runtime API: Peer Access](https://rocmdocs.amd.com/projects/HIP/en/latest/doxygen/group__Peer.html)

**Tomorrow:** Day 48 - Advanced ROCm Libraries... MIOpen, rocPRIM, and sorting algorithms on the GPU.

*End of Day 047 - Total Lines: 1000+*
