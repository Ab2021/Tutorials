# Lab 4.1: The OSI Model & TCP/IP

## 🎯 Objective

Demystify the 7 Layers of the OSI Model. Instead of just memorizing them, you will "see" them in action using packet analysis tools.

## 📋 Prerequisites

-   Linux Terminal.
-   `tcpdump` installed (`sudo apt install tcpdump`).

## 📚 Background

### The 7 Layers (Please Do Not Throw Sausage Pizza Away)

1.  **Physical**: Cables, WiFi signals. (Bits)
2.  **Data Link**: MAC Addresses, Switches. (Frames)
3.  **Network**: IP Addresses, Routers. (Packets) **<-- DevOps lives here**
4.  **Transport**: TCP/UDP, Ports. (Segments) **<-- And here**
5.  **Session**: Session management.
6.  **Presentation**: Encryption (SSL/TLS), Formatting.
7.  **Application**: HTTP, SSH, FTP. (Data) **<-- And here**

---

## 🔨 Hands-On Implementation

### Part 1: Layer 3 (Network) - IP 🌐

1.  **View your IP:**
    ```bash
    ip addr
    ```

2.  **Trace the route (Hops):**
    See every router between you and Google.
    ```bash
    traceroute google.com
    # OR
    tracepath google.com
    ```
    *Observation:* Each line is a Layer 3 hop.

### Part 2: Layer 4 (Transport) - TCP/UDP 🚚

1.  **Listen to traffic (`tcpdump`):**
    Open Terminal A:
    ```bash
    sudo tcpdump -i eth0 -n
    # Replace eth0 with your interface name (check ip addr)
    ```

2.  **Generate traffic:**
    Open Terminal B:
    ```bash
    curl http://example.com
    ```

3.  **Analyze Terminal A:**
    You will see the **3-Way Handshake**:
    -   `SYN` (Hello?)
    -   `SYN-ACK` (Hello, I hear you.)
    -   `ACK` (Great, let's talk.)

### Part 3: Layer 7 (Application) - HTTP 📄

1.  **Verbose Curl:**
    ```bash
    curl -v http://example.com
    ```
    *Output:*
    -   `> GET / HTTP/1.1` (Application Layer Request)
    -   `< HTTP/1.1 200 OK` (Application Layer Response)

---

## 🎯 Challenges

### Challenge 1: The Encapsulation (Difficulty: ⭐⭐)

**Task:**
Imagine sending a letter.
-   Letter content = Layer 7 (Data).
-   Envelope = Layer 4 (Port).
-   Mailbox Address = Layer 3 (IP).
-   Mail Truck = Layer 2 (MAC).
-   Road = Layer 1 (Physical).

**Question:** When a router receives a packet, which layers does it look at?
*Hint: Routers are "Layer 3 devices".*

### Challenge 2: Wireshark (Difficulty: ⭐⭐⭐)

**Task:**
If you have a GUI, install **Wireshark**. It's `tcpdump` with a UI.
1.  Capture traffic.
2.  Filter by `http`.
3.  Right-click a packet -> "Follow TCP Stream".
4.  Read the raw HTML of the website you visited.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Routers primarily look at **Layer 3 (IP Address)** to decide where to send the packet next. They strip off Layer 2 (MAC) and replace it with the MAC of the next hop.

**Challenge 2:**
(Wireshark is a GUI tool, no command solution).
</details>

---

## 🔑 Key Takeaways

1.  **Encapsulation**: Each layer wraps the previous one.
2.  **Troubleshooting**:
    -   Is cable plugged in? (Layer 1)
    -   Can I ping gateway? (Layer 2/3)
    -   Is port 80 open? (Layer 4)
    -   Is Nginx returning 500 error? (Layer 7)

---

## ⏭️ Next Steps

We know how data moves. Now let's learn how addresses work.

Proceed to **Lab 4.2: IP Addressing & Subnetting**.
