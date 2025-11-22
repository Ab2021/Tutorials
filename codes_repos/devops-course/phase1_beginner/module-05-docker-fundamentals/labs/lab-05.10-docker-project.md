# Lab 5.10: Docker Capstone Project

## 🎯 Objective

Containerize a complete **MERN Stack** (MongoDB, Express, React, Node) application - or a simplified version of it. You will write Dockerfiles for Frontend and Backend, and orchestrate them with Docker Compose.

## 📋 Prerequisites

-   Completed Module 5.
-   Docker & Docker Compose.

## 📚 Background

### The Architecture
1.  **Frontend**: React App (Served by Nginx).
2.  **Backend**: Node.js API.
3.  **Database**: MongoDB.

---

## 🔨 Hands-On Implementation

### Step 1: The Backend (Node.js) 🟢

1.  **Create `backend/server.js`:**
    ```javascript
    const express = require('express');
    const mongoose = require('mongoose');
    const app = express();

    mongoose.connect('mongodb://mongo:27017/mydb', { useNewUrlParser: true })
      .then(() => console.log('MongoDB Connected'))
      .catch(err => console.log(err));

    app.get('/', (req, res) => res.send('Hello from Backend!'));

    app.listen(5000, () => console.log('Server running on port 5000'));
    ```

2.  **Create `backend/Dockerfile`:**
    ```dockerfile
    FROM node:14
    WORKDIR /app
    COPY package*.json ./
    RUN npm install
    COPY . .
    CMD ["node", "server.js"]
    ```

### Step 2: The Frontend (React/Static) ⚛️

*For simplicity, we will use a static HTML file served by Nginx, simulating a build.*

1.  **Create `frontend/index.html`:**
    ```html
    <h1>Hello from Frontend!</h1>
    ```

2.  **Create `frontend/Dockerfile`:**
    ```dockerfile
    FROM nginx:alpine
    COPY index.html /usr/share/nginx/html
    ```

### Step 3: Orchestration (Compose) 🎼

1.  **Create `docker-compose.yml`:**
    ```yaml
    version: '3.8'

    services:
      frontend:
        build: ./frontend
        ports:
          - "80:80"
        depends_on:
          - backend

      backend:
        build: ./backend
        ports:
          - "5000:5000"
        depends_on:
          - mongo

      mongo:
        image: mongo:4.4
        volumes:
          - mongo-data:/data/db

    volumes:
      mongo-data:
    ```

### Step 4: Launch 🚀

1.  **Run:**
    ```bash
    docker-compose up --build
    ```

2.  **Verify:**
    -   Frontend: `http://localhost`
    -   Backend: `http://localhost:5000`
    -   Logs: Check if "MongoDB Connected" appears.

---

## 🎯 Challenges

### Challenge 1: Hot Reload (Difficulty: ⭐⭐⭐)

**Task:**
Configure the Backend service so that if you edit `server.js`, the server restarts automatically inside the container.
*Hint: Use `nodemon` and Bind Mounts.*

### Challenge 2: Multi-Stage React Build (Difficulty: ⭐⭐⭐⭐)

**Task:**
If you know React, create a real `create-react-app`.
Write a Dockerfile that:
1.  Builds the React app (`npm run build`).
2.  Copies the `build/` folder to Nginx.
    *This is a true production pattern.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Compose:
```yaml
backend:
  volumes:
    - ./backend:/app
  command: npm install -g nodemon && nodemon server.js
```
</details>

---

## 🔑 Key Takeaways

1.  **Microservices**: You just built a microservices architecture.
2.  **Persistence**: MongoDB data survives restarts thanks to the volume.
3.  **Isolation**: The Frontend talks to the Backend via the internal network, not the public internet (in a real setup).

---

## ⏭️ Next Steps

**Congratulations!** You have completed Module 5: Docker Fundamentals.
You can now package any application.

Proceed to **Module 6: CI/CD Basics**.
