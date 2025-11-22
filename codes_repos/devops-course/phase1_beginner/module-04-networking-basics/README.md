# Networking Basics for DevOps

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of networking basics, including:
- **Models**: Understanding the OSI 7-Layer Model and TCP/IP Stack.
- **Addressing**: Mastering IPv4, Subnetting, and CIDR.
- **Protocols**: Deep dive into TCP, UDP, DNS, and HTTP/S.
- **Security**: Implementing Firewalls, SSL/TLS, and SSH.
- **Troubleshooting**: Using tools like `curl`, `dig`, `netstat`, and `tcpdump`.

---

## 📖 Theoretical Concepts

### 1. The OSI Model & TCP/IP

The **OSI (Open Systems Interconnection)** model describes how computer systems communicate. While OSI has 7 layers, the practical internet runs on the **TCP/IP** model (4 layers).

| Layer (OSI) | Function | Protocol Examples | DevOps Relevance |
| :--- | :--- | :--- | :--- |
| **7. Application** | User interface & Data | HTTP, SSH, DNS | Load Balancers, API Gateways |
| **6. Presentation** | Encryption & Formatting | SSL/TLS, JSON | Encoding, Serialization |
| **5. Session** | Connection management | Sockets | Session Stickiness |
| **4. Transport** | Reliable delivery | TCP, UDP | Ports, Retries, Timeouts |
| **3. Network** | Routing & Addressing | IP, ICMP | VPCs, Subnets, Routing Tables |
| **2. Data Link** | Physical addressing | Ethernet (MAC) | Virtual Interfaces (veth) |
| **1. Physical** | Cables & Signals | Fiber, WiFi | Data Center Cabling |

### 2. IP Addressing & CIDR

Every device needs an IP address.
- **IPv4**: 32-bit (e.g., `192.168.1.1`). Running out of addresses.
- **IPv6**: 128-bit (e.g., `2001:0db8::`). The future.

**CIDR (Classless Inter-Domain Routing):**
Used to define network ranges.
- `/32`: 1 IP (Host).
- `/24`: 256 IPs (Standard LAN). `192.168.1.0/24`.
- `/16`: 65,536 IPs (Large Network).
- `/0`: All IPs (`0.0.0.0/0`).

### 3. Core Protocols

#### TCP vs UDP
- **TCP (Transmission Control Protocol)**: Reliable. Connection-oriented (3-way handshake). Retransmits lost packets. Used for Web (HTTP), Email (SMTP), File Transfer (FTP).
- **UDP (User Datagram Protocol)**: Fast. Fire-and-forget. No guarantee of delivery. Used for Streaming, Gaming, DNS.

#### DNS (Domain Name System)
The phonebook of the internet. Translates `google.com` -> `142.250.190.46`.
- **A Record**: Hostname to IPv4.
- **AAAA Record**: Hostname to IPv6.
- **CNAME**: Alias (www -> root).
- **TTL (Time To Live)**: Caching duration.

#### HTTP/HTTPS
- **HTTP**: Cleartext. Port 80.
- **HTTPS**: Encrypted (TLS). Port 443.
- **Methods**: GET, POST, PUT, DELETE, PATCH.
- **Status Codes**: 2xx (Success), 3xx (Redirect), 4xx (Client Error), 5xx (Server Error).

### 4. Network Security

- **Firewalls**: Filter traffic based on rules (Allow Port 80, Deny All).
- **Security Groups (Cloud)**: Stateful firewalls attached to instances.
- **SSL/TLS**: Encrypts data in transit. Uses Public/Private key pairs (PKI).
- **SSH (Secure Shell)**: Encrypted remote access. Uses keys instead of passwords for automation.

---

## 🔧 Practical Examples

### Troubleshooting with CLI Tools

**1. Check Connectivity (Ping)**
```bash
ping google.com
```

**2. Check DNS (Dig)**
```bash
dig google.com +short
```

**3. Check Ports (Netstat/SS)**
```bash
# List listening ports
ss -tuln
```

**4. Test HTTP (Curl)**
```bash
# Get headers only
curl -I https://google.com
```

**5. Capture Traffic (Tcpdump)**
```bash
# Capture port 80 traffic
sudo tcpdump -i eth0 port 80
```

---

## 🎯 Hands-on Labs

- [Lab 4.1: OSI Model & TCP/IP](./labs/lab-04.1-osi-tcpip.md)
- [Lab 4.10: Networking Capstone](./labs/lab-04.10-networking-project.md)
- [Lab 4.2: IP Addressing & Subnetting](./labs/lab-04.2-ip-subnetting.md)
- [Lab 4.3: DNS & Name Resolution](./labs/lab-04.3-dns-basics.md)
- [Lab 4.4: HTTP & HTTPS Protocols](./labs/lab-04.4-http-https.md)
- [Lab 4.5: SSH & Remote Access](./labs/lab-04.5-ssh-remote-access.md)
- [Lab 4.6: Firewalls & Security Groups](./labs/lab-04.6-firewalls.md)
- [Lab 4.7: Load Balancing Basics](./labs/lab-04.7-load-balancing.md)
- [Lab 4.8: Troubleshooting Tools (curl, dig, netstat)](./labs/lab-04.8-troubleshooting.md)
- [Lab 4.9: SSL/TLS & Certificates](./labs/lab-04.9-ssl-tls.md)

---

## 📚 Additional Resources

### Official Documentation
- [Mozilla Developer Network (MDN) - Networking](https://developer.mozilla.org/en-US/docs/Web/HTTP)
- [RFC Editor (The Standards)](https://www.rfc-editor.org/)

### Interactive Tutorials
- [Mess with DNS](https://messwithdns.net/)
- [CIDR.xyz](https://cidr.xyz/) - Visual Subnet Calculator.

---

## 🔑 Key Takeaways

1.  **It's Always DNS**: When something breaks, check DNS first.
2.  **Least Privilege**: Only open the ports you absolutely need (e.g., 80, 443, 22).
3.  **Latency Matters**: The speed of light is finite. CDN and Region selection impact performance.
4.  **Encryption Everywhere**: Never send sensitive data over HTTP.

---

## ⏭️ Next Steps

1.  Complete the labs to practice using these tools.
2.  Proceed to **[Module 5: Docker Fundamentals](../module-05-docker-fundamentals/README.md)** to start building containers.
