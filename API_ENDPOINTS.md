# 📚 Meetup Bot API Endpoints Documentation

## Base URL
```
Production: https://your-domain.com
Development: http://localhost:8000
```

## 🔥 Main Endpoints

### 1. **Get Event Recommendations** (Primary Endpoint)
```http
POST /api/recommendations
```

**Request Body:**
```json
{
  "query": "I want to play cricket this weekend",
  "user_id": "user123",              // Optional
  "user_current_city": "noida",      // IMPORTANT: Filters events by this city
  "limit": 5,                        // Optional, default 5
  "preferences": {                   // Optional, overrides saved preferences
    "activities": ["cricket", "football"],
    "areas": ["sector 62", "sector 18"],
    "budget_max": 500
  }
}
```

**Response:**
```json
{
  "success": true,
  "query": "I want to play cricket this weekend",
  "user_id": "user123",
  "user_current_city": "noida",
  "recommendations": [
    {
      "event_id": "evt_001",
      "name": "Weekend Cricket Tournament",
      "club_name": "Noida Sports Club",
      "activity": "cricket",
      "start_time": "2025-01-15 09:00 IST",
      "end_time": "2025-01-15 18:00 IST",
      "location": {
        "venue": "Noida Stadium",
        "area": "Sector 21",
        "city": "Noida",
        "full_address": "Noida Stadium, Sector 21, Noida"
      },
      "price": 300,
      "available_spots": 20,
      "registration_url": "https://register.example.com/evt_001"
    }
  ],
  "total_found": 8,
  "user_preferences_used": {...},
  "message": "Found 5 events in noida",
  "bot_response": "Great news! I found 5 amazing events for you in Noida..."
}
```

### 2. **Chat with Bot** (Natural Language + Structured Data)
```http
POST /api/chat
```

**Request Body:** (Same as /api/recommendations)

**Response:**
```json
{
  "success": true,
  "structured_data": {
    // Full recommendations response
  },
  "chat_response": "Hey! I found some amazing cricket events in Noida for you...",
  "combined_response": {
    "message": "Natural language response",
    "events": [...]  // Top 3 events
  }
}
```

## 📊 Monitoring Endpoints

### 3. **System Statistics**
```http
GET /api/stats
```

**Response:**
```json
{
  "events": {
    "collection_name": "meetup_events",
    "total_events": 1250,
    "host": "chromadb.misfits.net.in",
    "port": 8000,
    "last_updated": "2025-01-10T10:30:00"
  },
  "user_preferences": {
    "total_user_preferences": 450
  },
  "system_status": "operational"
}
```

### 4. **Sync Events** (Admin Only)
```http
POST /api/sync/events?full_sync=false
```

**Query Parameters:**
- `full_sync`: boolean (default: false) - Whether to do full sync or incremental

**Response:**
```json
{
  "success": true,
  "message": "Sync completed. Total events: 1255",
  "stats": {...}
}
```

## 🏥 Health Check Endpoints

### 5. **Root Health Check**
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Meetup Recommendation API",
  "total_events": 1250,
  "chroma_host": "chromadb.misfits.net.in",
  "chroma_port": 8000
}
```

### 6. **Kubernetes Liveness Probe**
```http
GET /health/liveness
```

**Response:**
```json
{
  "status": "alive"
}
```

### 7. **Kubernetes Readiness Probe**
```http
GET /health/readiness
```

**Response:**
```json
{
  "status": "ready",
  "events": 1250
}
```

## 🔑 Key Features

### City Filtering Logic
- **STRICT FILTERING**: When `user_current_city` is provided, ONLY events from that city are returned
- **Smart City Matching**: Handles city variations automatically
  - Gurugram ↔ Gurgaon
  - Delhi ↔ New Delhi
  - Bangalore ↔ Bengaluru
  - Mumbai ↔ Bombay
  - And more...

### Supported Cities
The API works with ANY city but has special handling for:
- Delhi / New Delhi
- Noida
- Gurugram / Gurgaon
- Mumbai / Bombay
- Bangalore / Bengaluru
- Chennai
- Kolkata
- Pune
- Hyderabad
- Ahmedabad
- Faridabad
- Ghaziabad
- And any other city in your database!

### Match Scoring
Events are internally scored and sorted based on:
- **City Match**: 35% weight (highest priority)
- **Activity Match**: 30% weight
- **Area Match**: 15% weight
- **Budget Match**: 15% weight
- **Availability**: 5% weight

*Note: Match scores and recommendation reasons are not included in the API response for cleaner, focused data.*

## 🚀 Quick Start Examples

### Example 1: User in Noida
```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "query": "weekend sports activities",
    "user_current_city": "noida",
    "limit": 3
  }'
```

### Example 2: User with Preferences
```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to play badminton",
    "user_id": "user456",
    "user_current_city": "delhi",
    "preferences": {
      "activities": ["badminton"],
      "budget_max": 500
    }
  }'
```

### Example 3: Chat Endpoint
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What fun activities are happening today?",
    "user_current_city": "mumbai"
  }'
```

## 📝 Important Notes

1. **Registration URLs**: Always included in responses. Falls back to "Contact organizer" if no URL available.

2. **City Filtering**: When `user_current_city` is provided, the API will ONLY return events from that city unless the user explicitly mentions another city in their query.

3. **Time Format**: All times are in IST (Indian Standard Time).

4. **User Preferences**: 
   - Saved preferences are loaded when `user_id` is provided
   - Can be overridden with the `preferences` field in the request

5. **Error Handling**: All endpoints return appropriate HTTP status codes and error messages.

## 🔒 Authentication
Currently no authentication required. In production, add API key authentication.

## 📈 Rate Limiting
No rate limiting in development. Production should implement rate limiting per IP/API key.