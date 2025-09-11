# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI Meetup Recommendation System** - An intelligent event recommendation platform using ChromaDB vector database for semantic search and personalized recommendations.

### Core Technologies
- **FastAPI** - Web framework for REST APIs and WebSocket support
- **ChromaDB** - Vector database (Remote server at 43.205.192.16:8000)
- **OpenAI/NVIDIA APIs** - LLM integration for conversational AI
- **Sentence Transformers** - Semantic embeddings for event matching
- **Docker/Kubernetes** - Container orchestration for deployment

## Architecture

### Directory Structure

```
Ai Agents/
├── model/                  # Production code
│   ├── ai-agent.py        # Main AI agent implementation
│   └── api_server.py      # FastAPI web server
├── learning/              # Learning/experimental code
│   ├── live-chorma.py     # Bot with remote ChromaDB
│   ├── model-with-upload.py  # File upload capabilities
│   ├── ai-agent-sync-data.py # Local ChromaDB version
│   ├── user-preference-with-csv.py # CSV preferences
│   └── last-with-anmol.py    # Previous iteration
├── Dockerfile             # Container configuration
├── kubernetes-deployment.yaml # K8s deployment
└── requirements.txt       # Python dependencies
```

### Component Structure

```
Remote ChromaDB (43.205.192.16:8000)
        ↑
        | HTTP API
        ↓
┌─────────────────────────┐
│  learning/live-chorma.py│ ← Bot with ChromaDBManager
└─────────────────────────┘
        ↑
        | Import
        ↓
┌─────────────────────────┐
│  model/api_server.py    │ ← FastAPI server (port 8000)
└─────────────────────────┘
        ↓
   REST/WebSocket
        ↓
   [Clients/UI]
```

### Production Files (model/)

- **model/ai-agent.py** - Main AI agent for production
  - Google Colab compatible
  - ChromaDB integration
  - NVIDIA API for LLM

- **model/api_server.py** - FastAPI web server
  - REST endpoints for recommendations
  - WebSocket for real-time chat
  - Health checks for Kubernetes
  - Currently imports from learning/live-chorma.py

### Learning/Experimental Files (learning/)

- **learning/live-chorma.py** - Bot with remote ChromaDB integration
  - ChromaDBManager class for vector operations
  - EventSync for API data synchronization
  - UserPreferenceManager for personalization
  - MeetupBot main orchestrator
  - Currently used by api_server.py

- **learning/model-with-upload.py** - Enhanced version with file upload
  - CSV upload for user preferences
  - Remote ChromaDB (43.205.192.16:8000)

- **learning/ai-agent-sync-data.py** - Local ChromaDB version
  - Original implementation with local storage
  - Used for development/testing

- **learning/user-preference-with-csv.py** - CSV-based preferences
  - User preference management via CSV

- **learning/last-with-anmol.py** - Previous iteration (backup)

## Development Commands

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run production API server (from model directory)
cd model
python api_server.py
# Server runs on http://localhost:8000

# Test API endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "find tech events", "user_id": "user123", "user_current_city": "Mumbai"}'
```

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Chat with user context (main endpoint)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "find sports events", "user_id": "user123", "user_current_city": "Mumbai"}'

# Check user preferences
curl http://localhost:8000/api/user/preferences/user123

# Save user preferences
curl -X POST http://localhost:8000/api/user/preferences \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "activities": ["sports", "tech"], "preferred_locations": ["Downtown"], "preferred_time": "evening"}'

# WebSocket connection
wscat -c ws://localhost:8000/ws/chat
```

### Data Synchronization

```bash
# Trigger event sync via API
curl -X POST http://localhost:8000/api/sync/events

# Check system stats
curl http://localhost:8000/api/stats
```

## Deployment

### Docker Build & Run

```bash
# Build Docker image
docker build -t misfits-event-recommendation:latest .

# Run container locally
docker run -d \
  -p 8000:8000 \
  -e CHROMA_HOST=43.205.192.16 \
  -e CHROMA_PORT=8000 \
  misfits-event-recommendation:latest
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes-deployment.yaml

# Check deployment status
kubectl get pods -l app=misfits-event-recommendation
kubectl get svc misfits-event-recommendation-service

# View logs
kubectl logs -l app=misfits-event-recommendation

# Scale deployment
kubectl scale deployment misfits-event-recommendation --replicas=5
```

### Environment Variables

```bash
# ChromaDB Connection (Remote Server)
CHROMA_HOST=43.205.192.16
CHROMA_PORT=8000

# API Keys
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_API_KEY=nvapi-[your-key]

# Model Settings
TRANSFORMERS_OFFLINE=1  # Use cached models
HF_HUB_OFFLINE=1       # Prevent model downloads
```

## ChromaDB Remote Server Setup

### Current Configuration
- **Host**: 43.205.192.16
- **Port**: 8000
- **Collections**: 
  - `meetup_events` - Event data with embeddings
  - `user_preferences` - User preference vectors

### Connecting to Remote ChromaDB

```python
from chromadb import HttpClient

# Initialize client
client = HttpClient(
    host="43.205.192.16",
    port=8000
)

# Or use ChromaDBManager
chroma_manager = ChromaDBManager(
    host="43.205.192.16",  # Defaults to this
    port=8000
)
```

### Migration from Local to Remote

The codebase has been migrated from local ChromaDB to remote server:
- Old: Local persistent directory (`./chroma_db/`)
- New: Remote HTTP API (43.205.192.16:8000)

Files already updated:
- `live-chorma.py` - Uses remote by default
- `model-with-upload.py` - Hardcoded remote server
- `last-with-anmol.py` - Hardcoded remote server

## API Endpoints

### Core Chat Endpoints

- `POST /api/chat` - Main chat endpoint with user context
- `POST /api/chat/simple` - Simplified chat (text only response)
- `WS /ws/chat` - WebSocket real-time chat

### User Preference Endpoints

- `GET /api/user/preferences/{user_id}` - Check if user has preferences
- `POST /api/user/preferences` - Save user preferences  
- `PUT /api/user/preferences/{user_id}` - Update user preferences

### System Endpoints

- `GET /` - Health check with stats
- `GET /health/liveness` - Kubernetes liveness probe
- `GET /health/readiness` - Kubernetes readiness probe
- `GET /api/stats` - System statistics
- `POST /api/sync/events` - Trigger event sync

### Request/Response Models

```python
# Chat Request (Required fields for user context)
{
    "message": str,
    "user_id": str,  # Required
    "user_current_city": str  # Required
}

# Chat Response
{
    "success": bool,
    "message": str,
    "events": [EventRecommendation],
    "total_found": int,
    "needs_preferences": bool,  # Triggers preference collection
    "user_preferences_used": dict
}

# User Preference Request
{
    "user_id": str,
    "activities": ["sports", "tech", "music"],
    "preferred_locations": ["Downtown", "North Zone"],
    "preferred_time": "evening",
    "budget_range": "500-1000"
}

# Event Recommendation
{
    "event_id": str,
    "name": str,
    "club_name": str,
    "activity": str,
    "start_time": str,
    "location": {
        "venue": str,
        "area": str,
        "city": str
    },
    "price": float,
    "registration_url": str
}
```

## Testing & Debugging

### ChromaDB Connection Test

```python
# Test remote connection
import chromadb
client = chromadb.HttpClient(host="43.205.192.16", port=8000)
print(client.list_collections())
```

### Common Issues

1. **ChromaDB Connection Failed**
   - Check if remote server is accessible
   - Verify CHROMA_HOST and CHROMA_PORT env vars
   - Test with curl: `curl http://43.205.192.16:8000`

2. **Model Download Issues**
   - Set TRANSFORMERS_OFFLINE=1 to use cached models
   - Pre-download models in Docker build

3. **Memory Issues in Kubernetes**
   - Adjust resource limits in kubernetes-deployment.yaml
   - Current: 512Mi-1Gi memory, 250m-500m CPU

## Code Patterns

### Working with ChromaDBManager

```python
# Always use remote server configuration
chroma_manager = ChromaDBManager(
    host="43.205.192.16",  # Remote server
    port=8000
)

# Search events
results = chroma_manager.search_events(
    query_text="tech meetups",
    n_results=5
)

# Add events
chroma_manager.add_events(events_df)
```

### Event Data Structure

```python
class EventDetailsForAgent:
    event_id: str
    event_name: str
    activity: str
    start_time: datetime
    location_name: str
    city_name: str
    club_name: str
    ticket_price: float
    available_spots: int
    event_url: str
```

## Performance Optimization

### Caching Strategy
- ChromaDB handles vector caching
- Use collection metadata for quick lookups
- Implement local result caching for frequent queries

### Scaling Considerations
- Kubernetes HPA configured (2-10 replicas)
- CPU target: 70% utilization
- Memory target: 80% utilization
- LoadBalancer service on port 8001

## Security Notes

### API Keys Management
- NVIDIA API key hardcoded (should move to secrets)
- No authentication on ChromaDB currently
- Consider implementing API key authentication for production

### Network Security
- ChromaDB server exposed publicly (43.205.192.16)
- Consider VPN or private network for production
- Implement rate limiting on API endpoints

## Future Improvements

### Planned Enhancements
1. Complete ChromaDB server authentication
2. Implement Redis caching layer
3. Add comprehensive logging/monitoring
4. Create admin dashboard for event management
5. Implement user authentication system

### Technical Debt
- Move hardcoded API keys to environment/secrets
- Add comprehensive error handling
- Implement retry logic for ChromaDB connections
- Add unit and integration tests
- Create CI/CD pipeline

## Application Integration Guide

### Chat Integration Flow

1. **User clicks chat button in your app**
2. **Pass user context automatically**:
   ```javascript
   const chatbot = new EventChatbot('http://api-server:8000', user.id, user.city);
   const response = await chatbot.sendMessage('find events for me');
   ```
3. **Handle preference collection**:
   - If `response.needs_preferences` is true, show preference dialog
   - Save preferences and retry the message
4. **Display events**: Show `response.events` in your UI

### Smart Conversation Handling

The bot automatically:
- **Understands activity requests**: "find sports events", "looking for tech meetups"
- **Uses user location**: Filters events by `user_current_city`
- **Manages preferences**: Asks for preferences only when needed
- **Provides personalized results**: Uses stored preferences for better recommendations

### Frontend Integration Examples

See `frontend-integration-examples.md` for complete JavaScript, React, and WebSocket examples.

## Quick Reference

### Essential Commands
```bash
# Start production API server
cd model && python api_server.py

# Docker build & run (production ready)
docker build -t misfits-event-recommendation . && docker run -p 8000:8000 misfits-event-recommendation

# Kubernetes deploy
kubectl apply -f kubernetes-deployment.yaml

# Check logs
kubectl logs -f -l app=misfits-event-recommendation

# Test health
curl http://localhost:8000/health

# Test chat with user context
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "find events", "user_id": "123", "user_current_city": "Mumbai"}'
```

### Production Files (model/)
- `model/ai-agent.py` - Complete bot with ChromaDB, user preferences, event sync
- `model/api_server.py` - Production FastAPI server with chat endpoints
- `Dockerfile` - Container setup using model/ directory
- `kubernetes-deployment.yaml` - Production K8s deployment
- `frontend-integration-examples.md` - Complete integration guide

### Key Features
- **User Context Injection**: Automatic user_id and location handling
- **Smart Preference Management**: Collects preferences when needed
- **Conversational AI**: Natural language understanding for event queries  
- **Scalable Deployment**: Kubernetes-ready with auto-scaling
- **Easy Integration**: Simple API calls from your application