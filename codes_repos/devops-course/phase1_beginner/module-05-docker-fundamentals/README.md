# Docker Fundamentals

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Docker, including:
- **Core Concepts**: Understanding how Containers differ from Virtual Machines.
- **Architecture**: The Docker Daemon, Client, Registry, and Image Layers.
- **Building**: Writing optimized `Dockerfiles` using Multi-Stage builds.
- **Running**: Managing container lifecycle, networking, and storage.
- **Orchestration**: Using Docker Compose for multi-container applications.

---

## 📖 Theoretical Concepts

### 1. Containerization vs Virtualization

- **Virtual Machines (VMs)**: Emulate hardware. Each VM has a full OS (Kernel + User Space). Heavy (GBs), slow boot (minutes).
- **Containers**: Share the Host OS Kernel. Isolate User Space (Binaries/Libs). Lightweight (MBs), fast boot (milliseconds).

**Why Docker?**
- **Consistency**: "It works on my machine" -> "It works everywhere."
- **Isolation**: Dependencies for App A don't conflict with App B.
- **Efficiency**: Higher density of applications per server.

### 2. Docker Architecture

- **Docker Daemon (`dockerd`)**: The background service that manages objects (images, containers, networks).
- **Docker Client (`docker`)**: The CLI tool you use. It talks to the Daemon via REST API.
- **Docker Registry**: Stores images (e.g., Docker Hub, AWS ECR).
- **Images**: Read-only templates. Built from layers (Union File System).
- **Containers**: Runnable instances of images. (Image + R/W Layer).

### 3. The Dockerfile

A text document that contains all the commands to assemble an image.

| Instruction | Purpose | Example |
| :--- | :--- | :--- |
| `FROM` | Base Image | `FROM python:3.9-slim` |
| `WORKDIR` | Set directory | `WORKDIR /app` |
| `COPY` | Copy files | `COPY . .` |
| `RUN` | Execute command (Build time) | `RUN pip install -r requirements.txt` |
| `CMD` | Default command (Run time) | `CMD ["python", "app.py"]` |
| `EXPOSE` | Document port | `EXPOSE 8080` |

### 4. Networking & Storage

#### Networking
- **Bridge (Default)**: Private network inside host. Containers talk via IP.
- **Host**: Container shares Host's IP stack. Fast, but port conflicts possible.
- **None**: No networking.

#### Storage
- **Volumes**: Managed by Docker (`/var/lib/docker/volumes`). Best for persistence.
- **Bind Mounts**: Maps a host file/folder to the container. Best for development (live reload).
- **Tmpfs**: Stored in memory. Lost on stop.

### 5. Docker Compose

A tool for defining and running multi-container Docker applications.
Uses a `docker-compose.yaml` file to configure services, networks, and volumes.

```yaml
version: '3.8'
services:
  web:
    build: .
    ports: ["5000:5000"]
  redis:
    image: "redis:alpine"
```

---

## 🔧 Practical Examples

### Basic Commands

```bash
# Run Nginx
docker run -d -p 8080:80 --name my-web nginx

# List running containers
docker ps

# Stop
docker stop my-web

# Remove
docker rm my-web
```

### Building an Image

```bash
# Build (tag it as 'my-app:v1')
docker build -t my-app:v1 .

# Run it
docker run my-app:v1
```

### Debugging

```bash
# View logs
docker logs -f my-web

# Enter container shell
docker exec -it my-web bash
```

---

## 🎯 Hands-on Labs

- [Lab 5.1: Introduction to Containers](./labs/lab-05.1-intro-containers.md)
- [Lab 5.10: Docker Capstone Project](./labs/lab-05.10-docker-project.md)
- [Lab 5.2: Docker Run & Basic Commands](./labs/lab-05.2-docker-run.md)
- [Lab 5.3: Dockerfiles & Building Images](./labs/lab-05.3-dockerfiles.md)
- [Lab 5.4: Docker Networking](./labs/lab-05.4-docker-networking.md)
- [Lab 5.5: Docker Volumes & Storage](./labs/lab-05.5-docker-volumes.md)
- [Lab 5.6: Docker Compose](./labs/lab-05.6-docker-compose.md)
- [Lab 5.7: Docker Best Practices (Security & Size)](./labs/lab-05.7-docker-best-practices.md)
- [Lab 5.8: Docker Registry & Hub](./labs/lab-05.8-docker-registry.md)
- [Lab 5.9: Debugging Containers](./labs/lab-05.9-debugging-containers.md)

---

## 📚 Additional Resources

### Official Documentation
- [Docker Documentation](https://docs.docker.com/)
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Compose File Reference](https://docs.docker.com/compose/compose-file/)

### Interactive Tutorials
- [Play with Docker](https://labs.play-with-docker.com/)
- [Katacoda (O'Reilly)](https://www.oreilly.com/online-learning/katacoda.html)

---

## 🔑 Key Takeaways

1.  **One Process Per Container**: Don't run SSH, Cron, and App in one container.
2.  **Ephemerality**: Containers should be disposable. Store data in Volumes.
3.  **Layer Caching**: Order matters in Dockerfile. Put frequently changing lines (Code) after stable lines (Dependencies).
4.  **Security**: Don't run as root. Use minimal base images (Alpine/Distroless).

---

## ⏭️ Next Steps

1.  Complete the labs to build your own containers.
2.  Proceed to **[Module 6: CI/CD Basics](../module-06-cicd-basics/README.md)** to automate the building of these containers.
