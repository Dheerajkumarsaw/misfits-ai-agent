"""
FastAPI Server for Meetup Recommendation Bot
Deployable on Kubernetes for scalable API access
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import uvicorn
import sys
import os

# Import the bot module from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MeetupBot from ai-agent.py in the same directory
print("🚀 Starting FastAPI server...")
print(f"📁 Working directory: {os.getcwd()}")
print(f"📁 Script directory: {os.path.dirname(__file__)}")

try:
    print("🔄 Attempting to import from ai_agent module...")
    from ai_agent import MeetupBot, ChromaDBManager
    print("✅ Successfully imported from ai_agent module")
except ImportError as e:
    print(f"⚠️  Direct import failed: {e}")
    print("🔄 Using fallback import method...")
    
    # Fallback import method
    import importlib.util
    ai_agent_path = os.path.join(os.path.dirname(__file__), "ai-agent.py")
    print(f"📁 Looking for ai-agent.py at: {ai_agent_path}")
    
    if not os.path.exists(ai_agent_path):
        print(f"❌ ai-agent.py not found at {ai_agent_path}")
        raise ImportError(f"Cannot find ai-agent.py at {ai_agent_path}")
    
    try:
        spec = importlib.util.spec_from_file_location("ai_agent", ai_agent_path)
        ai_agent = importlib.util.module_from_spec(spec)
        print("🔄 Executing ai-agent.py module...")
        spec.loader.exec_module(ai_agent)
        MeetupBot = ai_agent.MeetupBot
        ChromaDBManager = ai_agent.ChromaDBManager
        print("✅ Successfully imported using fallback method")
    except Exception as e:
        print(f"❌ Fallback import failed: {e}")
        raise

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
    # New fields from API
    activity_icon_url: Optional[str] = None
    club_icon_url: Optional[str] = None
    event_cover_image_url: Optional[str] = None
    event_uuid: Optional[str] = None
    participants_count: Optional[int] = 0


class ChatRequest(BaseModel):
    message: str
    user_id: str  # Required for internal app integration
    user_current_city: str  # Required for location-based recommendations

class ChatResponse(BaseModel):
    success: bool
    message: str
    events: List[EventRecommendation] = []
    total_found: int = 0
    needs_preferences: bool = False  # Flag to trigger preference collection

class UserPreferenceRequest(BaseModel):
    user_id: str
    activities: List[str]  # ["sports", "tech", "music", "arts"]
    preferred_locations: List[str]
    preferred_time: Optional[str] = None  # "morning", "evening", "weekend"
    budget_range: Optional[str] = None

class UserPreferenceResponse(BaseModel):
    success: bool
    message: str
    preferences: Optional[Dict[str, Any]] = None


class RecommendationResponse(BaseModel):
    success: bool
    recommendations: List[EventRecommendation]
    total_found: int
    message: str

# Initialize bot instance (singleton)
bot_instance = None

def get_bot():
    """Get or create bot instance"""
    global bot_instance
    if bot_instance is None:
        print("🔄 Creating MeetupBot instance...")
        try:
            bot_instance = MeetupBot()
            print("✅ Bot instance created successfully")
        except Exception as e:
            print(f"❌ Failed to create bot instance: {e}")
            raise
    return bot_instance

@app.on_event("startup")
async def startup_event():
    """Initialize bot on startup"""
    print("🚀 Starting Meetup Recommendation API...")
    try:
        bot = get_bot()
        stats = bot.chroma_manager.get_collection_stats()
        existing_events = stats.get('total_events', 0)
        print(f"✅ Bot initialized with {existing_events} existing events")
        
        # Determine if we need full sync (first time) or incremental sync
        if existing_events == 0:
            print("🔄 No existing events found. Running FULL initial sync...")
            print("📞 This will call /upcoming API to load all current events")
            bot.event_sync_manager.run_single_sync(full_sync=True)  # Calls /upcoming + /updated APIs
        else:
            print("🔄 Found existing events. Running incremental sync...")
            print("📞 This will call /updated API to get latest changes")
            bot.event_sync_manager.run_single_sync(full_sync=False)  # Only calls /updated API
        
        # Start periodic incremental sync (every 2 minutes) - only /updated API
        if not bot.event_sync_manager.is_running:
            bot.event_sync_manager.start_periodic_sync(interval_minutes=2)
            print("🔄 Started automatic incremental sync (/updated API every 2 minutes)")
        else:
            print("✅ Event sync is already running")
        
    except Exception as e:
        print(f"⚠️  Warning: Bot initialization failed: {str(e)}")
        print("⚠️  API will start but may not function properly until bot is initialized")
        # Don't fail startup - let the API start and handle errors per request

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🛑 Shutting down Meetup Recommendation API...")
    try:
        bot = get_bot()
        if bot.event_sync_manager.is_running:
            bot.event_sync_manager.stop_periodic_sync()
            print("✅ Stopped event sync")
    except Exception as e:
        print(f"⚠️ Error during shutdown: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    bot = get_bot()
    event_stats = bot.chroma_manager.get_collection_stats()
    user_stats = bot.chroma_manager.get_user_prefs_stats()
    
    return {
        "status": "healthy",
        "service": "Meetup Recommendation API",
        "total_events": event_stats.get('total_events', 0),
        "total_users": user_stats.get('total_user_preferences', 0),
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
            "limit": 5,
            "query": request.query,
            "user_current_city": request.user_current_city,
            "preferences": {}
        }
        
        # If current city is provided, add it to preferences for filtering
        if request.user_current_city:
            request_data["preferences"]["current_city"] = request.user_current_city
            
        # Get recommendations using the enhanced JSON method with extraction
        result = bot.get_recommendations_with_json_extraction(request_data)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to get recommendations"))
        
        # Format response for UI
        response = RecommendationResponse(
            success=True,
            recommendations=result["recommendations"],
            total_found=result["total_found"],
            message=result["message"]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Enhanced chat endpoint for internal application integration
    Automatically uses user context for personalized responses
    """
    try:
        bot = get_bot()

        message_lower = request.message.lower().strip()

        # Smart intent detection - understand what user really wants
        def detect_user_intent(message: str) -> str:
            """Detect user's actual intent from message
            Returns: 'search_events', 'restart', 'greeting', 'question', 'negative'
            """
            msg = message.lower().strip()

            # 1. Restart conversation intent
            restart_keywords = ["restart", "reset", "start over", "new conversation", "clear chat", "begin again", "fresh start"]
            if any(kw in msg for kw in restart_keywords):
                return 'restart'

            # 2. Negative intent - user explicitly doesn't want events
            negative_patterns = ["don't want", "not interested", "no thanks", "never mind", "don't show", "no event"]
            if any(pattern in msg for pattern in negative_patterns):
                return 'negative'

            # 3. Questions about bot capabilities
            question_patterns = ["what can you", "who are you", "what do you do", "how do you work", "tell me about"]
            if any(pattern in msg for pattern in question_patterns):
                return 'question'

            # 4. Event search intent - positive action words + event context
            search_patterns = ["find", "search", "show me", "looking for", "want to", "interested in",
                             "recommend", "suggest", "get me", "i need", "i want", "looking to", "looking up", "looking around", "looking out for", "hunt for", "seeking", "explore", "exploring", "browse", "browsing", "check out", "check for", "see if there are", "see if you can find"]
            event_words = ["event","events", "activity", "activities", "meetup", "meetups", "meet-up",  "meet-ups", "meet","meets" , "game", "games", "match", "matches", "session", "sessions", "class","classes", "workshop", "workshops",
            "jam", "jams",  "hike", "hikes" "hiking", "run", "runs", "running", "ride","rides", "cycling", "concert", "concerts", "festival", "festivals",
            "exhibition", "exhibitions", "conference", "conferences", "seminar", "seminars", "webinar", "webinars", "party", "parties", "social", "socials" "gathering", "gatherings", "outing", "outings", "show", "shows", "performance", "performances", "tournament", "tournaments",
             "hackathon", "hackathons", "bootcamp", "bootcamps", "retreat", "retreats", "networking", "network", "networks",
             "lecture", "lectures", "talk", "talks", "discussion", "discussions", "debate", "debates", "league", "leagues", "championship", "championships",
              "fitness", "workout", "workouts", "cooking", "cookings", "art", "arts", "craft", "crafts"]

            has_action = any(pattern in msg for pattern in search_patterns)
            has_event_context = any(word in msg for word in event_words)

            # Strong event intent if has action word OR event context with positive framing
            if has_action or has_event_context:
                return 'search_events'

            # 5. Default to greeting for casual conversation
            return 'greeting'

        # Detect the intent
        user_intent = detect_user_intent(message_lower)

        # Handle different intents
        if user_intent == 'restart':
            # Restart conversation - clear history and greet
            # Note: If you have conversation history tracking, clear it here
            greeting_response = bot.generate_personalized_greeting(
                user_id=request.user_id,
                include_event_teaser=False
            )
            return ChatResponse(
                success=True,
                message=f"🔄 Conversation restarted!\n\n{greeting_response}",
                events=[],
                total_found=0,
                needs_preferences=False
            )

        elif user_intent == 'negative':
            # User explicitly doesn't want events
            return ChatResponse(
                success=True,
                message="No problem! I'm here whenever you need help discovering events. Just let me know! 😊",
                events=[],
                total_found=0,
                needs_preferences=False
            )

        elif user_intent == 'question':
            # User asking about bot capabilities
            capabilities_message = """I'm Miffy, your friendly event discovery companion! 🌟

Here's what I can help you with:
• 🔍 Find events and activities based on your interests
• 🎯 Get personalized recommendations for meetups
• 📍 Discover events in your city
• 🎨 Explore various activities: sports, tech, music, arts, and more!

Just tell me what you're looking for, and I'll find the perfect events for you!"""
            return ChatResponse(
                success=True,
                message=capabilities_message,
                events=[],
                total_found=0,
                needs_preferences=False
            )

        elif user_intent == 'greeting':
            # Casual conversation without event intent
            greeting_response = bot.generate_personalized_greeting(
                user_id=request.user_id,
                include_event_teaser=False
            )
            return ChatResponse(
                success=True,
                message=greeting_response,
                events=[],
                total_found=0,
                needs_preferences=False
            )

        # Prepare request data with user context
        request_data = {
            "user_id": request.user_id,
            "limit": 5,
            "query": request.message,
            "user_current_city": request.user_current_city,
            "preferences": {"current_city": request.user_current_city}
        }

        # Get recommendations using the enhanced JSON method with extraction
        print(f"🔍 Chat API: Calling get_recommendations_with_json_extraction with: {request_data}")
        try:
            result = bot.get_recommendations_with_json_extraction(request_data)
            print(f"🔍 Chat API: Recommendations result success: {result.get('success', False)}")
            print(f"🔍 Chat API: Recommendations count: {len(result.get('recommendations', []))}")
        except Exception as e:
            print(f"❌ Chat API: Error in get_recommendations_with_json_extraction: {e}")
            import traceback
            traceback.print_exc()
            result = {"success": False, "recommendations": [], "total_found": 0, "message": str(e)}
        
        # Check if AI agent is requesting preferences
        if result.get("needs_preferences", False):
            return ChatResponse(
                success=True,
                message=result.get("message", "I'd love to help you find events! To give you personalized recommendations, could you tell me what activities and interests you enjoy?"),
                events=[],
                total_found=0,
                needs_preferences=True
            )
        
        # Generate clean, structured response
        events = result.get("recommendations", [])[:3]
        
        if result.get("success", False) and events:
            message = f"Found {len(events)} event{'s' if len(events) > 1 else ''} for you in {request.user_current_city}!"
            return ChatResponse(
                success=True,
                message=message,
                events=events,
                total_found=result["total_found"],
                needs_preferences=False
            )
        elif result.get("success", False):
            return ChatResponse(
                success=True,
                message=f"No events found matching your criteria in {request.user_current_city}. Try different keywords or check nearby cities.",
                events=[],
                total_found=result["total_found"],
                needs_preferences=False
            )
        else:
            # Use message from the result, with fallback to generic message
            message = result.get("message") or "I'm having trouble finding events right now. Please try again."
            return ChatResponse(
                success=False,
                message=message,
                events=[],
                total_found=result.get("total_found", 0),
                needs_preferences=False
            )
        
    except Exception as e:
        return ChatResponse(
            success=False,
            message="Sorry, I encountered an error processing your request. Please try again.",
            events=[],
            total_found=0,
            needs_preferences=False
        )

@app.post("/api/chat/simple")
async def simple_chat(request: ChatRequest):
    """
    Simplified chat endpoint returning just the conversational response
    Perfect for basic chat integration
    """
    try:
        bot = get_bot()
        
        # Get natural language response with user context  
        chat_response = bot.get_bot_response_json(request.message, request.user_id)
        
        # For queries about events, also get quick recommendations
        event_keywords = ["event", "find", "search", "recommend", "suggest", "looking for"]
        if any(keyword in request.message.lower() for keyword in event_keywords):
            request_data = {
                "user_id": request.user_id,
                "limit": 3,
                "query": request.message,
                "user_current_city": request.user_current_city,
                "preferences": {"current_city": request.user_current_city}
            }
            result = bot.get_recommendations_with_json_extraction(request_data)
            
            if result["success"] and result["recommendations"]:
                events_summary = f"\n\nI found {result['total_found']} events for you. Here are the top 3:\n"
                for i, event in enumerate(result["recommendations"][:3], 1):
                    # Safely access location fields
                    location_city = 'Unknown'
                    if isinstance(event.get('location'), dict):
                        location_city = event['location'].get('city', 'Unknown')
                    events_summary += f"{i}. {event['name']} - {location_city}\n"
                chat_response += events_summary
        
        return {
            "success": True,
            "message": chat_response,
            "user_id": request.user_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": "Sorry, I couldn't process your request right now. Please try again.",
            "user_id": request.user_id
        }

@app.get("/api/user/preferences/{user_id}", response_model=UserPreferenceResponse)
async def get_user_preferences(user_id: str):
    """
    Check if user has preferences stored in ChromaDB
    """
    try:
        bot = get_bot()
        user_prefs = bot.chroma_manager.get_user_preferences_by_user_id(user_id)
        
        if user_prefs:
            return UserPreferenceResponse(
                success=True,
                message="User preferences found",
                preferences=user_prefs
            )
        else:
            return UserPreferenceResponse(
                success=False,
                message="No preferences found for this user",
                preferences=None
            )
    except Exception as e:
        return UserPreferenceResponse(
            success=False,
            message=f"Error retrieving preferences: {str(e)}",
            preferences=None
        )

@app.post("/api/user/preferences", response_model=UserPreferenceResponse)
async def save_user_preferences(request: UserPreferenceRequest):
    """
    Save user preferences to ChromaDB
    """
    try:
        bot = get_bot()
        
        # Create preference document
        preference_doc = {
            "user_id": request.user_id,
            "activities": request.activities,
            "preferred_locations": request.preferred_locations,
            "preferred_time": request.preferred_time,
            "budget_range": request.budget_range,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to ChromaDB
        success = bot.chroma_manager.add_user_preferences_batch([preference_doc])
        
        if success:
            return UserPreferenceResponse(
                success=True,
                message="User preferences saved successfully",
                preferences=preference_doc
            )
        else:
            return UserPreferenceResponse(
                success=False,
                message="Failed to save user preferences",
                preferences=None
            )
    except Exception as e:
        return UserPreferenceResponse(
            success=False,
            message=f"Error saving preferences: {str(e)}",
            preferences=None
        )

@app.put("/api/user/preferences/{user_id}", response_model=UserPreferenceResponse)
async def update_user_preferences(user_id: str, request: UserPreferenceRequest):
    """
    Update existing user preferences in ChromaDB
    """
    try:
        bot = get_bot()
        
        # Create updated preference document
        preference_doc = {
            "user_id": user_id,
            "activities": request.activities,
            "preferred_locations": request.preferred_locations,
            "preferred_time": request.preferred_time,
            "budget_range": request.budget_range,
            "updated_at": datetime.now().isoformat()
        }
        
        # Update in ChromaDB (add will overwrite existing)
        success = bot.chroma_manager.add_user_preferences_batch([preference_doc])
        
        if success:
            return UserPreferenceResponse(
                success=True,
                message="User preferences updated successfully",
                preferences=preference_doc
            )
        else:
            return UserPreferenceResponse(
                success=False,
                message="Failed to update user preferences",
                preferences=None
            )
    except Exception as e:
        return UserPreferenceResponse(
            success=False,
            message=f"Error updating preferences: {str(e)}",
            preferences=None
        )

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    bot = get_bot()
    event_stats = bot.chroma_manager.get_collection_stats()
    user_prefs_stats = bot.chroma_manager.get_user_prefs_stats()
    
    return {
        "events": event_stats,
        "user_preferences": user_prefs_stats,
        "system_status": "operational",
        "sync_running": bot.event_sync_manager.is_running
    }

@app.get("/api/sync/status")
async def get_sync_status():
    """Get sync status"""
    bot = get_bot()
    return {
        "sync_running": bot.event_sync_manager.is_running,
        "sync_interval_minutes": 2,
        "updated_events_api": bot.event_sync_manager.updated_api_url,
        "upcoming_events_api": bot.event_sync_manager.upcoming_api_url
    }

@app.get("/api/cache/stats")
async def get_cache_stats():
    """
    Get cache performance statistics
    Shows hit rates, time saved, and cache efficiency metrics
    """
    bot = get_bot()
    return bot.get_cache_stats()

@app.post("/api/cache/clear")
async def clear_caches(user_id: Optional[str] = None):
    """
    Clear cache entries
    If user_id provided, clears only that user's cache
    Otherwise clears all caches
    """
    bot = get_bot()
    if user_id:
        bot.invalidate_user_cache(user_id)
        return {"success": True, "message": f"Cleared cache for user {user_id}"}
    else:
        bot.clear_all_caches()
        return {"success": True, "message": "Cleared all caches"}

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

@app.post("/api/sync/users")
async def sync_users():
    """Trigger user preference synchronization from API"""
    try:
        bot = get_bot()
        synced_count = bot.sync_users_once()
        return {
            "success": True,
            "message": f"User sync completed. Total synced: {synced_count}",
            "synced_count": synced_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sync/users/status")
async def get_user_sync_status():
    """Get user sync status"""
    try:
        bot = get_bot()
        status = bot.get_user_sync_status()
        return {
            "success": True,
            "status": status
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

# Realtime chat over WebSocket
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        bot = get_bot()
        await websocket.send_json({"type": "ready", "message": "WebSocket connected"})
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"query": raw}

            query = payload.get("query") or payload.get("message") or ""
            user_id = payload.get("user_id")
            user_current_city = payload.get("user_current_city") or payload.get("city")

            message_lower = query.lower().strip()

            # Smart intent detection - understand what user really wants
            def detect_user_intent(message: str) -> str:
                """Detect user's actual intent from message
                Returns: 'search_events', 'restart', 'greeting', 'question', 'negative'
                """
                msg = message.lower().strip()

                # 1. Restart conversation intent
                restart_keywords = ["restart", "reset", "start over", "new conversation", "clear chat", "begin again", "fresh start"]
                if any(kw in msg for kw in restart_keywords):
                    return 'restart'

                # 2. Negative intent - user explicitly doesn't want events
                negative_patterns = ["don't want", "not interested", "no thanks", "never mind", "don't show", "no event"]
                if any(pattern in msg for pattern in negative_patterns):
                    return 'negative'

                # 3. Questions about bot capabilities
                question_patterns = ["what can you", "who are you", "what do you do", "how do you work", "tell me about"]
                if any(pattern in msg for pattern in question_patterns):
                    return 'question'

                # 4. Event search intent - positive action words + event context
                search_patterns = ["find", "search", "show me", "looking for", "want to", "interested in",
                                 "recommend", "suggest", "get me", "i need", "i want"]
                event_words = ["event", "activity", "activities", "meetup", "game", "session", "class", "workshop"]

                has_action = any(pattern in msg for pattern in search_patterns)
                has_event_context = any(word in msg for word in event_words)

                # Strong event intent if has action word OR event context with positive framing
                if has_action or has_event_context:
                    return 'search_events'

                # 5. Default to greeting for casual conversation
                return 'greeting'

            # Detect the intent
            user_intent = detect_user_intent(message_lower)

            # Handle different intents
            if user_intent == 'restart':
                # Restart conversation - clear history and greet
                greeting_response = bot.generate_personalized_greeting(
                    user_id=user_id,
                    include_event_teaser=False
                )
                await websocket.send_json({
                    "type": "response",
                    "success": True,
                    "message": f"🔄 Conversation restarted!\n\n{greeting_response}",
                    "recommendations": [],
                    "total_found": 0
                })
                continue

            elif user_intent == 'negative':
                # User explicitly doesn't want events
                await websocket.send_json({
                    "type": "response",
                    "success": True,
                    "message": "No problem! I'm here whenever you need help discovering events. Just let me know! 😊",
                    "recommendations": [],
                    "total_found": 0
                })
                continue

            elif user_intent == 'question':
                # User asking about bot capabilities
                capabilities_message = """I'm Miffy, your friendly event discovery companion! 🌟

Here's what I can help you with:
• 🔍 Find events and activities based on your interests
• 🎯 Get personalized recommendations for meetups
• 📍 Discover events in your city
• 🎨 Explore various activities: sports, tech, music, arts, and more!

Just tell me what you're looking for, and I'll find the perfect events for you!"""
                await websocket.send_json({
                    "type": "response",
                    "success": True,
                    "message": capabilities_message,
                    "recommendations": [],
                    "total_found": 0
                })
                continue

            elif user_intent == 'greeting':
                # Casual conversation without event intent
                greeting_response = bot.generate_personalized_greeting(
                    user_id=user_id,
                    include_event_teaser=False
                )
                await websocket.send_json({
                    "type": "response",
                    "success": True,
                    "message": greeting_response,
                    "recommendations": [],
                    "total_found": 0
                })
                continue

            # Build recommendation request-compatible payload
            request_data = {
                "user_id": user_id,
                "query": query,
                "user_current_city": user_current_city,
                "preferences": {}
            }
            if user_current_city:
                request_data["preferences"]["current_city"] = user_current_city

            # Get structured recommendations
            print(f"🔍 WebSocket: Calling get_recommendations_with_json_extraction with: {request_data}")
            try:
                recommendations_result = bot.get_recommendations_with_json_extraction(request_data)
                print(recommendations_result)
                print(f"🔍 WebSocket: Recommendations result success: {recommendations_result.get('success', False)}")
                print(f"🔍 WebSocket: Recommendations count: {len(recommendations_result.get('recommendations', []))}")
                if not recommendations_result.get('success', False):
                    print(f"🔍 WebSocket: Recommendations error: {recommendations_result.get('message', 'Unknown error')}")
            except Exception as e:
                print(f"❌ WebSocket: Error calling get_recommendations_with_json_extraction: {e}")
                import traceback
                traceback.print_exc()
                recommendations_result = {"success": False, "recommendations": [], "total_found": 0, "message": str(e)}

            # Prepare clean structured response
            events = recommendations_result.get("recommendations", [])[:3]
            
            # Generate interactive response
            if events:
                message = f"Found {len(events)} event{'s' if len(events) > 1 else ''} for you!"
                success = True
            elif recommendations_result.get("success", False):
                message = recommendations_result.get("message", "No events found matching your criteria.")
                success = False  # No events found
            else:
                message = recommendations_result.get("message", "No events found.")
                success = False

            response_data = {
                "type": "response", 
                "success": success,
                "message": message,
                "recommendations": events,
                "total_found": recommendations_result.get("total_found", 0)
            }
            
            await websocket.send_json(response_data)
    except WebSocketDisconnect:
        # Client disconnected
        return
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "success": False,
                "error": str(e),
                "message": "An error occurred while processing your message."
            })
        except Exception:
            pass

if __name__ == "__main__":
    # Run with uvicorn for production
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )