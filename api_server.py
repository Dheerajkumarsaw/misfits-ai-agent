"""
FastAPI Server for Meetup Recommendation Bot
Deployable on Kubernetes for scalable API access
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import uvicorn
import sys
import os

# Import the bot module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Note: We'll import the bot after ensuring the module is loaded
# from live_chorma import MeetupBot

app = FastAPI(
    title="Meetup Recommendation API",
    description="AI-powered personalized event recommendations",
    version="1.0.0"
)

# Add CORS middleware for web UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class RecommendationRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    user_current_city: Optional[str] = None
    limit: Optional[int] = 5
    preferences: Optional[Dict[str, Any]] = None

class EventLocation(BaseModel):
    venue: str
    area: str
    city: str
    full_address: Optional[str] = None

class EventRecommendation(BaseModel):
    event_id: str
    name: str
    club_name: str
    activity: str
    start_time: str
    end_time: str
    location: EventLocation
    price: float
    available_spots: int
    registration_url: str

class RecommendationResponse(BaseModel):
    success: bool
    query: str
    user_id: Optional[str]
    user_current_city: Optional[str]
    recommendations: List[EventRecommendation]
    total_found: int
    message: str
    user_preferences_used: Optional[Dict[str, Any]] = None
    bot_response: Optional[str] = None  # Natural language response for UI

# Initialize bot instance (singleton)
bot_instance = None

def get_bot():
    """Get or create bot instance"""
    global bot_instance
    if bot_instance is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("live_chorma", os.path.join(os.path.dirname(__file__), "live-chorma.py"))
        live_chorma = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live_chorma)
        MeetupBot = live_chorma.MeetupBot
        bot_instance = MeetupBot()
        # Skip initial sync during startup - let it happen on first request if needed
        print("✅ Bot instance created successfully")
    return bot_instance

@app.on_event("startup")
async def startup_event():
    """Initialize bot on startup"""
    print("🚀 Starting Meetup Recommendation API...")
    try:
        bot = get_bot()
        stats = bot.chroma_manager.get_collection_stats()
        print(f"✅ Bot initialized with {stats.get('total_events', 0)} events")
    except Exception as e:
        print(f"⚠️  Warning: Bot initialization failed: {str(e)}")
        print("⚠️  API will start but may not function properly until bot is initialized")
        # Don't fail startup - let the API start and handle errors per request

@app.get("/")
async def root():
    """Health check endpoint"""
    bot = get_bot()
    stats = bot.chroma_manager.get_collection_stats()
    return {
        "status": "healthy",
        "service": "Meetup Recommendation API",
        "total_events": stats.get('total_events', 0),
        "chroma_host": bot.chroma_manager.host,
        "chroma_port": bot.chroma_manager.port
    }

@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized event recommendations based on query and user preferences
    """
    try:
        bot = get_bot()
        
        # Prepare enhanced request data
        request_data = {
            "user_id": request.user_id,
            "limit": request.limit or 5,
            "query": request.query,
            "user_current_city": request.user_current_city,
            "preferences": request.preferences or {}
        }
        
        # If current city is provided, add it to preferences for filtering
        if request.user_current_city:
            request_data["preferences"]["current_city"] = request.user_current_city
            
        # Get recommendations using the enhanced JSON method
        result = bot.get_recommendations_json(request_data)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to get recommendations"))
        
        # Format response for UI
        response = RecommendationResponse(
            success=True,
            query=request.query,
            user_id=request.user_id,
            user_current_city=request.user_current_city,
            recommendations=result["recommendations"],
            total_found=result["total_found"],
            message=result["message"],
            user_preferences_used=result.get("user_preferences_used"),
            bot_response=result.get("bot_response")  # Natural language response
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_bot(request: RecommendationRequest):
    """
    Chat endpoint that returns both structured data and natural language response
    """
    try:
        bot = get_bot()
        
        # Get structured recommendations
        recommendations_result = await get_recommendations(request)
        
        # Also get natural language response
        chat_response = bot.get_bot_response_json(request.query, request.user_id)
        
        return {
            "success": True,
            "structured_data": recommendations_result.dict(),
            "chat_response": chat_response,
            "combined_response": {
                "message": chat_response,
                "events": recommendations_result.recommendations[:3]  # Top 3 for chat
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Sorry, I couldn't process your request. Please try again."
        }

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    bot = get_bot()
    event_stats = bot.chroma_manager.get_collection_stats()
    user_prefs_stats = bot.chroma_manager.get_user_prefs_stats()
    
    return {
        "events": event_stats,
        "user_preferences": user_prefs_stats,
        "system_status": "operational"
    }

@app.post("/api/sync/events")
async def sync_events(full_sync: bool = False):
    """Trigger event synchronization"""
    try:
        bot = get_bot()
        bot.sync_events_once(full_sync=full_sync)
        stats = bot.chroma_manager.get_collection_stats()
        return {
            "success": True,
            "message": f"Sync completed. Total events: {stats.get('total_events', 0)}",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Kubernetes health check endpoints
@app.get("/health/liveness")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive"}

@app.get("/health/readiness")
async def readiness():
    """Kubernetes readiness probe"""
    try:
        bot = get_bot()
        # Just check if bot can be created - don't require events
        return {"status": "ready", "bot_initialized": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Bot initialization failed: {str(e)}")

if __name__ == "__main__":
    # Run with uvicorn for production
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )