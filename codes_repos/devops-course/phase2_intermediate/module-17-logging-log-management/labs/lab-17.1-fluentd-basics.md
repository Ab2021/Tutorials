# Lab 17.1: Fluentd Basics

## 🎯 Objective

Unified Logging Layer. Applications log in different formats (JSON, Text, Apache). **Fluentd** acts as a converter, taking logs from anywhere, parsing them, and sending them to anywhere (S3, Elasticsearch, Slack).

## 📋 Prerequisites

-   Docker installed.

## 📚 Background

### Architecture
-   **Input**: Where logs come from (Tail file, HTTP, Syslog).
-   **Filter**: Modify/Parse logs (Grep, Record Transformer).
-   **Output**: Where logs go (Stdout, File, S3, ES).
-   **Tag**: How Fluentd routes logs (e.g., `app.backend` vs `app.frontend`).

---

## 🔨 Hands-On Implementation

### Part 1: Configuration ⚙️

1.  **Create `fluent.conf`:**
    ```apache
    # Input: Receive logs via HTTP
    <source>
      @type http
      port 9880
      bind 0.0.0.0
    </source>

    # Filter: Add a field "host"
    <filter test.tag>
      @type record_transformer
      <record>
        hostname "#{Socket.gethostname}"
        processed_by "fluentd"
      </record>
    </filter>

    # Output: Print to Stdout
    <match test.tag>
      @type stdout
    </match>
    ```

### Part 2: Run Fluentd 🐳

1.  **Run Container:**
    ```bash
    docker run -d \
      -p 9880:9880 \
      -v $(pwd)/fluent.conf:/fluentd/etc/fluent.conf \
      --name fluentd \
      fluent/fluentd:v1.14-1
    ```

2.  **Check Logs:**
    ```bash
    docker logs -f fluentd
    ```

### Part 3: Send Logs 📨

1.  **Send JSON via Curl:**
    ```bash
    curl -X POST -d 'json={"message":"hello world"}' \
      http://localhost:9880/test.tag
    ```

2.  **Verify:**
    Check the docker logs. You should see:
    ```json
    {
      "message": "hello world",
      "hostname": "...",
      "processed_by": "fluentd"
    }
    ```
    *Note:* The `hostname` and `processed_by` fields were added by the filter!

### Part 4: Parsing (Regex) 🧩

Real logs are messy. Let's parse a fake Apache log.

1.  **Update `fluent.conf`:**
    ```apache
    <source>
      @type dummy
      tag dummy.apache
      dummy {"log": "127.0.0.1 - - [10/Oct/2000:13:55:36 -0700] \"GET /apache_pb.gif HTTP/1.0\" 200 2326"}
    </source>

    <filter dummy.apache>
      @type parser
      key_name log
      <parse>
        @type apache2
      </parse>
    </filter>

    <match dummy.apache>
      @type stdout
    </match>
    ```

2.  **Restart Fluentd:**
    ```bash
    docker restart fluentd
    ```

3.  **Observe:**
    The raw string is now a structured JSON object:
    `{"host": "127.0.0.1", "method": "GET", "code": "200", ...}`

---

## 🎯 Challenges

### Challenge 1: Output to File (Difficulty: ⭐⭐)

**Task:**
Change the output plugin to write logs to a file inside the container (`/fluentd/log/test.log`).
*Hint:* `@type file`.

### Challenge 2: Routing (Difficulty: ⭐⭐⭐)

**Task:**
Send logs with tag `app.error` to one file, and `app.info` to another.
*Hint:* Use two `<match>` blocks with different patterns.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
```apache
<match app.error>
  @type file
  path /fluentd/log/error
</match>

<match app.info>
  @type file
  path /fluentd/log/info
</match>
```
</details>

---

## 🔑 Key Takeaways

1.  **Decoupling**: Your app shouldn't know about S3 or Elasticsearch. It should just log to stdout, and Fluentd handles the rest.
2.  **Buffering**: Fluentd buffers logs in memory/file. If the destination (S3) is down, Fluentd retries later. No data loss.
3.  **Plugins**: Fluentd has 1000+ plugins.

---

## ⏭️ Next Steps

Fluentd is great for moving logs. But how do we query them cheaply?

Proceed to **Lab 17.2: Loki & Grafana (PLG Stack)**.
