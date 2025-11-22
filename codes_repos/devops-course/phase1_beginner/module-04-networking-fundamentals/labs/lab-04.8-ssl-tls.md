# Lab 4.8: SSL/TLS & Certificates

## 🎯 Objective

Understand Encryption in Transit. You will generate a Self-Signed Certificate, configure Nginx to use HTTPS, and analyze the certificate chain.

## 📋 Prerequisites

-   OpenSSL installed.
-   Nginx installed (or Docker).

## 📚 Background

### HTTP vs HTTPS
-   **HTTP**: Plain text. Anyone on the WiFi can read your passwords.
-   **HTTPS**: Encrypted (TLS). Only you and the server can read it.

### The Certificate Authority (CA)
Who do you trust?
-   Your browser trusts a list of Root CAs (DigiCert, Let's Encrypt).
-   The CA signs the Server's Certificate.
-   If the signature matches, the browser shows the Green Lock 🔒.

---

## 🔨 Hands-On Implementation

### Part 1: Generate Private Key & CSR 🔑

1.  **Generate Private Key:**
    ```bash
    openssl genrsa -out server.key 2048
    ```
    *Note:* Keep this safe!

2.  **Generate CSR (Certificate Signing Request):**
    ```bash
    openssl req -new -key server.key -out server.csr
    ```
    *Prompts:* Country, State, Common Name (CN).
    *CN is crucial:* It must match your domain (e.g., `localhost` or `mysite.com`).

### Part 2: Generate Self-Signed Cert 📜

Since we don't want to pay a CA, we will sign it ourselves.

1.  **Sign it:**
    ```bash
    openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
    ```
    *Result:* You now have `server.crt` (Public) and `server.key` (Private).

### Part 3: Configure Nginx for HTTPS 🔒

1.  **Create Config:**
    ```nginx
    events {}
    http {
        server {
            listen 443 ssl;
            server_name localhost;
            
            ssl_certificate /etc/nginx/certs/server.crt;
            ssl_certificate_key /etc/nginx/certs/server.key;
            
            location / {
                return 200 "Hello Secure World!";
            }
        }
    }
    ```

2.  **Run Nginx (Docker):**
    ```bash
    docker run -d -p 443:443 \
      -v $(pwd)/server.crt:/etc/nginx/certs/server.crt \
      -v $(pwd)/server.key:/etc/nginx/certs/server.key \
      -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf \
      nginx
    ```

3.  **Test:**
    ```bash
    curl -k https://localhost
    ```
    *Note:* `-k` tells curl to ignore the fact that it's self-signed (Insecure).

### Part 4: Inspecting Certs 🧐

1.  **Check Expiry Date:**
    ```bash
    openssl x509 -in server.crt -text -noout | grep "Not After"
    ```

2.  **Check Remote Site:**
    ```bash
    echo | openssl s_client -connect google.com:443 2>/dev/null | openssl x509 -noout -dates
    ```

---

## 🎯 Challenges

### Challenge 1: The Browser Warning (Difficulty: ⭐)

**Task:**
Open `https://localhost` in Chrome/Firefox.
What happens? Why?
Click "Advanced" -> "Proceed" to bypass it.

### Challenge 2: Let's Encrypt (Difficulty: ⭐⭐⭐⭐)

**Task:**
Research **Certbot**.
If you have a real domain and a public server, try to get a *real* free certificate using Let's Encrypt.
*Note: This requires a public IP, so you might not be able to do it on a local lab.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Browser shows "Your connection is not private".
Reason: The certificate is signed by "You", and the browser doesn't trust "You". It only trusts Root CAs.

**Challenge 2:**
```bash
sudo apt install certbot
sudo certbot --nginx
```
</details>

---

## 🔑 Key Takeaways

1.  **Private Key Protection**: If a hacker gets `server.key`, they can decrypt your traffic.
2.  **Self-Signed**: Good for testing/internal tools. Bad for public websites.
3.  **Expiry**: Certs expire (usually 90 days or 1 year). Monitoring expiry is a critical DevOps task.

---

## ⏭️ Next Steps

We have covered the protocols. Now let's look at the modern way to run networks.

Proceed to **Lab 4.9: VPC & Cloud Networking**.
