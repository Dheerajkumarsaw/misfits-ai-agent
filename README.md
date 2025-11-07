# AI Meetup Recommendation System - Production API

This project implements an intelligent meetup recommendation system using ChromaDB for vector storage, FastAPI for REST/WebSocket APIs, and NVIDIA's LLM for conversational AI.

## 🚀 Features

- **Vector-based Event Search**: ChromaDB for semantic similarity search
- **Personalized Recommendations**: User preference management and context-aware suggestions
- **Conversational AI**: Natural language chat interface powered by NVIDIA API
- **Real-time Communication**: WebSocket support for live chat
- **RESTful APIs**: Complete CRUD operations for events and user preferences
- **Auto-sync**: Automatic event synchronization from external APIs
- **Production Ready**: Kubernetes deployment with health checks and auto-scaling

## 📁 Project Structure

```
Ai Agents/
├── model/                      # Production code
│   ├── ai-agent.py            # Main bot implementation
│   └── api_server.py          # FastAPI web server
├── learning/                   # Experimental/development code
├── Dockerfile                  # Container configuration
├── kubernetes-deployment.yaml  # K8s deployment config
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.9+**
- **pip** package manager
- **ChromaDB Server** running at `43.205.192.16:8000` (already configured)

### 1. Virtual Environment Setup

#### **Mac/Linux**

```bash
# Navigate to project directory
cd "Ai Agents"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show venv path)
which python
```

#### **Windows**

```cmd
# Navigate to project directory
cd "Ai Agents"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (should show venv path)
where python
```

### 2. Install Dependencies

#### **For API Server** (Main Application)

```bash
# With venv activated, install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep fastapi
pip list | grep chromadb
```

#### **For ChromaDB Server** (If running your own ChromaDB instance)

```bash
# Install ChromaDB server
pip install chromadb

# Run ChromaDB server
chroma run --host 0.0.0.0 --port 8000

# Or with Docker
docker run -d -p 8000:8000 chromadb/chroma
```

**Note**: The project is configured to use a remote ChromaDB server at `43.205.192.16:8000`. You don't need to run a local ChromaDB server unless you want to test locally.

### 3. Running the Production API Server

#### **Mac/Linux**

```bash
# Navigate to model directory
cd model

# Run the FastAPI server
python3 api_server.py
```

#### **Windows**

```cmd
# Navigate to model directory
cd model

# Run the FastAPI server
python api_server.py
```

**Expected Output:**
```
🚀 Starting FastAPI server...
✅ Successfully imported from ai_agent module
🚀 Starting Meetup Recommendation API...
✅ Bot initialized with X existing events
🔄 Started automatic incremental sync
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The server will start on **http://localhost:8000**

## 🧪 API Testing

### Health Check

```bash
# Check if server is running
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","service":"Meetup Recommendation API","total_events":150,"total_users":10,...}
```

### Chat API (Main Endpoint)

```bash
# Chat with user context
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "find tech events",
    "user_id": "user123",
    "user_current_city": "Mumbai"
  }'

# Expected response:
# {
#   "success": true,
#   "message": "Found 3 events for you in Mumbai!",
#   "events": [...],
#   "total_found": 15,
#   "needs_preferences": false
# }
```

### User Preferences API

```bash
# Get user preferences
curl http://localhost:8000/api/user/preferences/user123

# Save user preferences
curl -X POST http://localhost:8000/api/user/preferences \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "activities": ["sports", "tech"],
    "preferred_locations": ["Downtown", "North Zone"],
    "preferred_time": "evening",
    "budget_range": "500-1000"
  }'
```

### Event Sync API

```bash
# Trigger event synchronization
curl -X POST http://localhost:8000/api/sync/events

# Get sync status
curl http://localhost:8000/api/sync/status
```

## 🌐 WebSocket Testing

### Method 1: Using wscat (Terminal)

#### **Installation**

```bash
# Mac/Linux
npm install -g wscat

# Windows
npm install -g wscat
```

#### **Testing WebSocket**

```bash
# Connect to WebSocket
wscat -c ws://localhost:8000/ws/chat

# After connection, send messages in JSON format:
{"message": "find sports events", "user_id": "user123", "user_current_city": "Mumbai"}

# Or simple text (query only):
{"query": "tech meetups in Mumbai", "user_id": "user123", "city": "Mumbai"}

# Expected response:
# {
#   "type": "response",
#   "success": true,
#   "message": "Found 3 events for you!",
#   "recommendations": [...],
#   "total_found": 10
# }
```

### Method 2: Using Postman

1. **Create New WebSocket Request**
   - Click **New** → **WebSocket Request**
   - Enter URL: `ws://localhost:8000/ws/chat`
   - Click **Connect**

2. **Send Messages**
   - In the message input area, select **JSON** format
   - Paste the following:
   ```json
   {
     "message": "find tech events",
     "user_id": "user123",
     "user_current_city": "Mumbai"
   }
   ```
   - Click **Send**

3. **View Responses**
   - Real-time responses appear in the **Messages** panel
   - Each response includes event recommendations

4. **Example Messages to Test**

   ```json
   // General event search
   {
     "message": "find events near me",
     "user_id": "dev_user_001",
     "user_current_city": "Delhi"
   }

   // Activity-specific search
   {
     "query": "football events this weekend",
     "user_id": "sports_fan_42",
     "city": "Bangalore"
   }

   // With preferences context
   {
     "message": "recommend something fun",
     "user_id": "user123",
     "user_current_city": "Mumbai"
   }
   ```

### Method 3: Browser JavaScript Console

```javascript
// Open browser console (F12) and paste:
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
  console.log('✅ Connected to WebSocket');
  ws.send(JSON.stringify({
    message: "find tech events",
    user_id: "browser_user",
    user_current_city: "Pune"
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('📩 Response:', data);
};

ws.onerror = (error) => {
  console.error('❌ WebSocket error:', error);
};
```

### Method 4: Python Script

```python
import websocket
import json

def on_message(ws, message):
    print(f"📩 Received: {message}")

def on_open(ws):
    print("✅ WebSocket connected")
    ws.send(json.dumps({
        "message": "find sports events",
        "user_id": "python_user",
        "user_current_city": "Chennai"
    }))

ws = websocket.WebSocketApp(
    "ws://localhost:8000/ws/chat",
    on_message=on_message,
    on_open=on_open
)

ws.run_forever()
```

## 🗄️ API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check with stats |
| `GET` | `/health/liveness` | Kubernetes liveness probe |
| `GET` | `/health/readiness` | Kubernetes readiness probe |
| `POST` | `/api/chat` | Main chat endpoint (recommended) |
| `POST` | `/api/chat/simple` | Simplified text-only chat |
| `WS` | `/ws/chat` | WebSocket real-time chat |
| `GET` | `/api/user/preferences/{user_id}` | Get user preferences |
| `POST` | `/api/user/preferences` | Save user preferences |
| `PUT` | `/api/user/preferences/{user_id}` | Update user preferences |
| `POST` | `/api/sync/events` | Trigger event sync |
| `GET` | `/api/sync/status` | Get sync status |
| `GET` | `/api/stats` | System statistics |

## 🐛 Troubleshooting

### Virtual Environment Issues

#### **Mac/Linux**
```bash
# If venv activation fails
deactivate  # Exit current venv
rm -rf venv  # Remove corrupted venv
python3 -m venv venv  # Recreate
source venv/bin/activate

# Permission denied error
chmod +x venv/bin/activate
```

#### **Windows**
```cmd
# If activation fails with script execution policy error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use alternative activation
venv\Scripts\activate.bat
```

### Port Already in Use

```bash
# Mac/Linux - Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Windows - Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### ChromaDB Connection Issues

```bash
# Test ChromaDB server connectivity
curl http://43.205.192.16:8000/api/v1/heartbeat

# If running local ChromaDB
chroma run --host 0.0.0.0 --port 8000
```

### Dependency Errors

```bash
# Upgrade pip
pip install --upgrade pip

# Force reinstall all dependencies
pip install --force-reinstall -r requirements.txt

# Install specific problematic package
pip install --no-cache-dir chromadb
```

### WebSocket Connection Failed

```bash
# Check if server is running
curl http://localhost:8000/health

# Check WebSocket endpoint is accessible
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  http://localhost:8000/ws/chat
```

## ⚙️ Environment Configuration

### Optional Environment Variables

```bash
# ChromaDB Configuration
export CHROMA_HOST=43.205.192.16
export CHROMA_PORT=8000

# Model Settings (for offline mode)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# API Keys (if using different LLM providers)
export OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export OPENAI_API_KEY=nvapi-your-key-here
```

### Running with Custom Configuration

```bash
# Mac/Linux
CHROMA_HOST=localhost CHROMA_PORT=8000 python3 api_server.py

# Windows
set CHROMA_HOST=localhost && set CHROMA_PORT=8000 && python api_server.py
```

## 🚢 Docker Deployment

### Build and Run with Docker

```bash
# Build Docker image
docker build -t misfits-event-recommendation:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e CHROMA_HOST=43.205.192.16 \
  -e CHROMA_PORT=8000 \
  --name event-api \
  misfits-event-recommendation:latest

# View logs
docker logs -f event-api

# Stop container
docker stop event-api
```

## ☸️ Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes-deployment.yaml

# Check deployment
kubectl get pods -l app=misfits-event-recommendation
kubectl get svc misfits-event-recommendation-service

# View logs
kubectl logs -l app=misfits-event-recommendation -f

# Scale deployment
kubectl scale deployment misfits-event-recommendation --replicas=5

# Delete deployment
kubectl delete -f kubernetes-deployment.yaml
```

## 📖 Quick Reference

### Complete Setup Commands (Copy-Paste)

#### **Mac/Linux**
```bash
# Clone or navigate to project
cd "Ai Agents"

# Setup venv and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run API server
cd model
python3 api_server.py
```

#### **Windows**
```cmd
# Clone or navigate to project
cd "Ai Agents"

# Setup venv and install
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run API server
cd model
python api_server.py
```

## 📄 License

This project is open source and available under the MIT License.

---

**For more detailed information**, see `CLAUDE.md` for development guidelines and architecture details. 