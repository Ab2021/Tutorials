# Lab 4.3: DNS & HTTP

## 🎯 Objective

Understand the two most important protocols for the web. You will manually query DNS servers to understand record types and construct raw HTTP requests to understand headers and status codes.

## 📋 Prerequisites

-   `dig` (from `dnsutils` or `bind-utils`).
-   `curl` and `telnet` (or `nc`).

## 📚 Background

### DNS (Domain Name System)
The phonebook of the internet.
-   **A Record**: Name -> IPv4 (`google.com` -> `1.2.3.4`).
-   **AAAA Record**: Name -> IPv6.
-   **CNAME**: Alias (`www.google.com` -> `google.com`).
-   **MX**: Mail server.
-   **TXT**: Text (used for verification/SPF).

### HTTP (HyperText Transfer Protocol)
The language of the web.
-   **Request**: Method (GET/POST), Path, Headers, Body.
-   **Response**: Status Code (200, 404, 500), Headers, Body.

---

## 🔨 Hands-On Implementation

### Part 1: DNS Deep Dive (`dig`) 📒

1.  **Standard Query (A Record):**
    ```bash
    dig google.com
    ```
    *Look for:* `ANSWER SECTION`.

2.  **Specific Record Type (MX):**
    Find out who handles email for Google.
    ```bash
    dig google.com MX
    ```

3.  **Trace the Hierarchy (`+trace`):**
    Watch the query go from Root (.) -> TLD (.com) -> Google Nameserver.
    ```bash
    dig google.com +trace
    ```

4.  **Reverse Lookup (`-x`):**
    IP -> Name.
    ```bash
    dig -x 8.8.8.8
    ```
    *Result:* `dns.google`.

### Part 2: HTTP Anatomy (`curl` & `telnet`) 📄

1.  **Verbose Curl:**
    ```bash
    curl -v https://httpbin.org/get
    ```
    *Observe:* The handshake (TLS) and the headers (`User-Agent`, `Accept`).

2.  **Manual HTTP Request (The Hard Way):**
    Use `telnet` or `nc` to type raw HTTP.
    ```bash
    telnet google.com 80
    ```
    *Type quickly:*
    ```http
    GET / HTTP/1.1
    Host: google.com

    ```
    (Press Enter twice).
    *Result:* You get the raw HTML response.

3.  **Status Codes:**
    -   **200 OK**: Success.
    -   **301 Moved**: Redirect (Location header tells where).
    -   **404 Not Found**: Client error (Bad URL).
    -   **500 Server Error**: Server crashed.
    -   **502 Bad Gateway**: Upstream server failed (common in Nginx/Load Balancers).

### Part 3: Troubleshooting 🔧

**Scenario:** A user says "The website is down".

1.  **Check DNS:**
    `dig website.com` -> Do we get an IP?
2.  **Check TCP:**
    `nc -zv website.com 80` -> Is port open?
3.  **Check HTTP:**
    `curl -I website.com` -> Is status 200?

---

## 🎯 Challenges

### Challenge 1: Spoofing DNS (Difficulty: ⭐⭐)

**Task:**
Force `curl` to think `example.com` is hosted on `1.1.1.1` (Cloudflare).
*Hint: `curl --resolve` or edit `/etc/hosts`.*

### Challenge 2: Inspecting Headers (Difficulty: ⭐⭐)

**Task:**
Use `curl` to find out what software `google.com` is running (Server header).
*Note: Many modern sites hide this for security.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
curl --resolve example.com:80:1.1.1.1 http://example.com
```
*Result:* You will get a 404 or 403 from Cloudflare because Cloudflare doesn't know what to do with a request for "example.com" on that IP.

**Challenge 2:**
```bash
curl -I google.com
# Look for "Server: gws" (Google Web Server)
```
</details>

---

## 🔑 Key Takeaways

1.  **It's always DNS**: If you can ping the IP but not the name, it's DNS.
2.  **Headers matter**: Authentication tokens, caching rules, and content types all live in headers.
3.  **502 vs 500**: 500 means the app crashed. 502 means the Load Balancer can't reach the app.

---

## ⏭️ Next Steps

We can talk to servers. Now let's log into them securely.

Proceed to **Lab 4.4: SSH & Secure Access**.
