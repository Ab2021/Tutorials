# Lab 5.6: Docker Compose

## 🎯 Objective

Orchestrate multi-container applications. Instead of running 5 long `docker run` commands, you will define your entire stack (App + DB + Cache) in a single YAML file.

## 📋 Prerequisites

-   Completed Lab 5.5.
-   Docker Compose installed (Included in Docker Desktop).

## 📚 Background

### Infrastructure as Code (for Local Dev)
`docker-compose.yml` describes:
1.  **Services**: The containers (App, DB).
2.  **Networks**: How they talk.
3.  **Volumes**: Where they store data.

**Commands:**
-   `docker-compose up`: Start everything.
-   `docker-compose down`: Stop and remove everything.

---

## 🔨 Hands-On Implementation

### Part 1: The Stack 🥞

We will build a **Python Web App** that uses **Redis** for a counter.

1.  **Create `app.py`:**
    ```python
    from flask import Flask
    from redis import Redis

    app = Flask(__name__)
    redis = Redis(host='redis', port=6379)

    @app.route('/')
    def hello():
        count = redis.incr('hits')
        return f'Hello! I have been seen {count} times.\n'

    if __name__ == "__main__":
        app.run(host="0.0.0.0", debug=True)
    ```

2.  **Create `Dockerfile`:**
    ```dockerfile
    FROM python:3.9-slim
    WORKDIR /code
    RUN pip install flask redis
    COPY app.py .
    CMD ["python", "app.py"]
    ```

### Part 2: The Compose File 📝

1.  **Create `docker-compose.yml`:**
    ```yaml
    version: '3.8'
    
    services:
      web:
        build: .
        ports:
          - "5000:5000"
        depends_on:
          - redis
          
      redis:
        image: "redis:alpine"
        volumes:
          - redis-data:/data

    volumes:
      redis-data:
    ```

### Part 3: Launch 🚀

1.  **Start:**
    ```bash
    docker-compose up
    # OR detached
    docker-compose up -d
    ```

2.  **Test:**
    Visit `http://localhost:5000`.
    Refresh. The counter goes up.

3.  **Inspect:**
    ```bash
    docker-compose ps
    ```
    *Note:* It created a network automatically (`lab56_default`) and DNS works (`web` can talk to `redis` by name).

4.  **Stop:**
    ```bash
    docker-compose down
    ```
    *Note:* This removes containers and networks, but **keeps volumes** by default.

---

## 🎯 Challenges

### Challenge 1: Environment Variables (Difficulty: ⭐⭐)

**Task:**
1.  Modify `app.py` to print a background color from `os.environ['BG_COLOR']`.
2.  Add `environment:` section to `docker-compose.yml` to set `BG_COLOR=blue`.
3.  Restart (`up -d`) and verify.

### Challenge 2: Scaling (Difficulty: ⭐⭐⭐)

**Task:**
Run 3 copies of the `web` service.
1.  Remove `ports: "5000:5000"` (because you can't bind 3 containers to port 5000).
2.  Add an Nginx Load Balancer service to the compose file to distribute traffic.
3.  Run `docker-compose up --scale web=3`.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
YAML:
```yaml
web:
  environment:
    - BG_COLOR=blue
```

**Challenge 2:**
This requires advanced Nginx config, but the command is:
```bash
docker-compose up -d --scale web=3
```
</details>

---

## 🔑 Key Takeaways

1.  **One Command**: `docker-compose up` is all a new developer needs to run your entire complex app.
2.  **Networking**: Compose handles network creation and DNS automatically.
3.  **Not for Prod**: While possible, Docker Compose is typically used for Development and Testing. For Production, use Kubernetes (Module 12).

---

## ⏭️ Next Steps

We have a running stack. How do we make the image smaller and safer?

Proceed to **Lab 5.7: Docker Best Practices**.
