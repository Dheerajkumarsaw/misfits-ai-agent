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

# Import new modules for enhancements
try:
    from cache_manager import ServerCache
    from session_manager import SessionManager
    from background_jobs import BackgroundJobs
    print("✅ Enhancement modules imported successfully")
except ImportError as e:
    print(f"⚠️ Failed to import enhancement modules: {e}")
    print("⚠️ Server will run without caching and session features")
    ServerCache = None
    SessionManager = None
    BackgroundJobs = None

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

# Initialize enhancement instances (global singletons)
cache_instance = None
session_manager_instance = None
background_jobs_instance = None

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

def get_cache():
    """Get or create cache instance"""
    global cache_instance
    if cache_instance is None and ServerCache is not None:
        cache_instance = ServerCache(max_size=1000)
    return cache_instance

def get_session_manager():
    """Get or create session manager instance"""
    global session_manager_instance
    if session_manager_instance is None and SessionManager is not None:
        session_manager_instance = SessionManager()
    return session_manager_instance

def get_background_jobs():
    """Get or create background jobs instance"""
    global background_jobs_instance
    if background_jobs_instance is None and BackgroundJobs is not None:
        bot = get_bot()
        background_jobs_instance = BackgroundJobs(bot)
    return background_jobs_instance

def detect_activities_in_query(query: str) -> List[str]:
    """
    Detect activity keywords in user query

    Args:
        query: User's search query

    Returns:
        List of detected activities
    """
    # Comprehensive activity keywords with typo variations (39 activity types from all_activities_db)
    activity_keywords = {
        # Sports - Ball Games
        'football': ['football', 'footballs', 'soccer', 'futsal', 'footbal', 'foot ball'],
        'cricket': ['cricket', 'crickets', 'criket', 'crikett'],
        'box_cricket': ['box cricket', 'indoor cricket', 'box criket'],
        'badminton': ['badminton', 'badmintons', 'badmington', 'badminten', 'badmintom'],
        'basketball': ['basketball', 'basketballs', 'hoops', 'basket ball', 'basketbal'],
        'volleyball': ['volleyball', 'volleyballs', 'voleyball', 'vollyball', 'volley ball'],
        'pickleball': ['pickleball', 'pickleballs', 'pickle ball', 'pickle balls', 'pickeball', 'picklebll', 'pickle-ball'],
        'bowling': ['bowling', 'bowl', 'bowlin'],

        # Sports - Fitness & Outdoor
        'running': ['running', 'run', 'runs', 'marathon', 'marathons', 'sprint', 'sprints', 'jogging', 'runing'],
        'cycling': ['cycling', 'bike', 'bikes', 'biking', 'bicycle', 'bicycles', 'cylcing', 'cyceling', 'cyclying'],
        'yoga': ['yoga', 'yogas', 'fitness', 'yogo', 'youga'],
        'hiking': ['hiking', 'hike', 'hikes', 'trek', 'treks', 'trekking', 'hikin', 'hikeing'],

        # Arts & Creative
        'dance': ['dance', 'dances', 'dancing', 'zumba', 'salsa', 'hip hop', 'hiphop', 'danc'],
        'music': ['music', 'concert', 'concerts', 'jam', 'jams', 'band', 'bands', 'singing', 'jam session', 'jam sessions', 'musik', 'musix'],
        'art': ['art', 'arts', 'painting', 'paintings', 'drawing', 'drawings', 'craft', 'crafts'],
        'photography': ['photography', 'photo', 'photos', 'photoshoot', 'photoshoots', 'photograpy', 'photograhpy', 'photografy'],
        'writing': ['writing', 'creative writing', 'writer', 'writers', 'writting', 'writng'],
        'poetry': ['poetry', 'poems', 'poem', 'open mic', 'open mics', 'spoken word', 'poetri', 'poitry'],

        # Entertainment & Social
        'boardgaming': ['board game', 'board games', 'boardgame', 'boardgames', 'tabletop', 'tabletops',
                       'borad game', 'borad games', 'bored game', 'bored games', 'bord game', 'bord games'],
        'video_games': ['gaming', 'game', 'games', 'esports', 'esport', 'video game', 'video games', 'videogame', 'videogames', 'gamming', 'gameing'],
        'chess': ['chess', 'ches'],
        'drama': ['drama', 'dramas', 'theater', 'theaters', 'theatre', 'theatres', 'acting', 'play', 'plays', 'drame'],
        'films': ['movie', 'movies', 'film', 'films', 'cinema', 'cinemas', 'flim', 'flims'],
        'quiz': ['trivia', 'trivias', 'quiz', 'quizzes', 'quizz', 'quizes'],
        'book_club': ['book club', 'book clubs', 'bookclub', 'bookclubs', 'reading', 'book', 'books', 'book-club'],
        'social_deductions': ['social deduction', 'social deductions', 'social deducation', 'social deducations',
                             'deduction', 'deductions', 'deducation', 'deducations',
                             'mafia', 'werewolf', 'werewolves', 'among us', 'amoung us'],

        # Professional & Community
        'community_space': ['tech', 'techs', 'coding', 'programming', 'hackathon', 'hackathons', 'hackaton', 'hackathn',
                           'developer', 'developers', 'bootcamp', 'bootcamps', 'boot camp', 'boot-camp',
                           'software', 'networking', 'meetup', 'meetups', 'professional', 'professionals',
                           'startup', 'startups', 'start up', 'start-up',
                           'entrepreneur', 'entrepreneurs', 'entrepeneur', 'entreprenuer',
                           'business', 'businesses', 'entrepreneurship', 'community', 'community space', 'technology'],
        'content_creation': ['content creation', 'content creator', 'creators', 'influencer', 'influencers', 'creator', 'content-creation'],

        # Lifestyle & Wellness
        'food': ['food', 'foods', 'cooking', 'culinary', 'baking', 'cookin'],
        'mindfulness': ['mindfulness', 'meditation', 'wellness', 'mental health', 'wellbeing',
                       'mindfullness', 'mindfullnes', 'mindfullnss', 'mindfulnes'],
        'journaling': ['journaling', 'journal', 'journals', 'diary', 'journalling'],
        'inner_journey': ['inner journey', 'spiritual', 'spirituality', 'personal growth', 'self discovery', 'self-discovery', 'inner-journey'],

        # Special Interest
        'harry_potter': ['harry potter', 'hp', 'potterhead', 'potterheads', 'hogwarts', 'harry poter', 'harrypotter', 'harry-potter'],
        'pop_culture': ['pop culture', 'popculture', 'pop', 'fandom', 'fandoms', 'pop-culture', 'popcultre'],
        'travel': ['travel', 'travels', 'trip', 'trips', 'adventure', 'adventures', 'explore', 'exploration', 'travle', 'trvel'],

        # Multi-purpose
        'multi_activity_club': ['multi activity', 'multi-activity', 'mixed activities', 'various activities', 'multi activity club'],
        'hni': ['hni', 'high net worth', 'premium', 'exclusive', 'luxury'],
        'others': ['others', 'misc', 'miscellaneous', 'general', 'various']
    }

    query_lower = query.lower()
    detected = []

    for activity, keywords in activity_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected.append(activity)

    return detected


def map_activity_to_db_type(activity_names: List[str]) -> List[str]:
    """
    Map detected activity names to database activity types

    EXACT MAPPING to all_activities_db from ai_agent.py:
    ['ART', 'BADMINTON', 'BASKETBALL', 'BOARDGAMING', 'BOOK_CLUB', 'BOWLING', 'BOX_CRICKET',
     'CHESS', 'COMMUNITY_SPACE', 'CONTENT_CREATION', 'CYCLING', 'DANCE', 'DEFAULT', 'DRAMA',
     'FILMS', 'FOOD', 'FOOTBALL', 'HARRY_POTTER', 'HIKING', 'HNI', 'INNER_JOURNEY',
     'JOURNALING', 'MINDFULNESS', 'MULTI_ACTIVITY_CLUB', 'MUSIC', 'OTHERS', 'PHOTOGRAPHY',
     'PICKLEBALL', 'POETRY', 'POP_CULTURE', 'QUIZ', 'RUNNING', 'SOCIAL_DEDUCTIONS',
     'TRAVEL', 'VIDEO_GAMES', 'VOLLEYBALL', 'WRITING', 'YOGA']

    Args:
        activity_names: List of detected activity names from query

    Returns:
        List of database activity type strings
    """
    # Mapping from detected keywords to exact database activity field values
    activity_mapping = {
        # Sports - Ball Games
        'football': 'FOOTBALL',
        'cricket': 'CRICKET',
        'box_cricket': 'BOX_CRICKET',
        'badminton': 'BADMINTON',
        'basketball': 'BASKETBALL',
        'volleyball': 'VOLLEYBALL',
        'pickleball': 'PICKLEBALL',
        'bowling': 'BOWLING',

        # Sports - Fitness & Outdoor
        'running': 'RUNNING',
        'cycling': 'CYCLING',
        'yoga': 'YOGA',
        'hiking': 'HIKING',

        # Arts & Creative
        'dance': 'DANCE',
        'music': 'MUSIC',
        'art': 'ART',
        'photography': 'PHOTOGRAPHY',
        'writing': 'WRITING',
        'poetry': 'POETRY',

        # Entertainment & Social
        'boardgaming': 'BOARDGAMING',
        'video_games': 'VIDEO_GAMES',
        'chess': 'CHESS',
        'drama': 'DRAMA',
        'films': 'FILMS',
        'quiz': 'QUIZ',
        'book_club': 'BOOK_CLUB',
        'social_deductions': 'SOCIAL_DEDUCTIONS',

        # Professional & Community
        'community_space': 'COMMUNITY_SPACE',
        'content_creation': 'CONTENT_CREATION',

        # Lifestyle & Wellness
        'food': 'FOOD',
        'mindfulness': 'MINDFULNESS',
        'journaling': 'JOURNALING',
        'inner_journey': 'INNER_JOURNEY',

        # Special Interest
        'harry_potter': 'HARRY_POTTER',
        'pop_culture': 'POP_CULTURE',
        'travel': 'TRAVEL',

        # Multi-purpose
        'multi_activity_club': 'MULTI_ACTIVITY_CLUB',
        'hni': 'HNI',
        'others': 'OTHERS'
    }

    db_types = []
    for activity in activity_names:
        activity_lower = activity.lower()
        if activity_lower in activity_mapping:
            db_types.append(activity_mapping[activity_lower])

    return db_types


def validate_event_activity(event: dict, required_activities: List[str]) -> bool:
    """
    Validate if event's activity matches the required activities

    Args:
        event: Event dictionary with 'activity' field
        required_activities: List of required DB activity types (e.g., ["BOARDGAMING"])

    Returns:
        True if event matches, False otherwise
    """
    if not required_activities:
        return True  # No filtering if no specific activity detected

    event_activity = event.get('activity', '').upper()
    return event_activity in required_activities


def detect_date_filter(query: str) -> Optional[Dict]:
    """
    Detect date-related keywords in query (uses IST timezone)

    Args:
        query: User's search query

    Returns:
        Dictionary with date filter info or None
        Format: {'type': 'today'|'tomorrow'|'weekend'|'this_week', 'dates': [date objects]}
    """
    from datetime import datetime, timedelta, timezone

    query_lower = query.lower()

    # Get current time in IST (UTC+5:30)
    ist_offset = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist_offset)

    # Today / Tonight
    if 'today' in query_lower or 'tonight' in query_lower:
        today = now_ist.date()
        return {'type': 'today', 'dates': [today]}

    # Tomorrow (with typo support)
    if 'tomorrow' in query_lower or 'tommorow' in query_lower or 'tommorrow' in query_lower:
        tomorrow = (now_ist + timedelta(days=1)).date()
        return {'type': 'tomorrow', 'dates': [tomorrow]}

    # This weekend
    if 'this weekend' in query_lower or 'weekend' in query_lower:
        # Weekend = Saturday and Sunday
        days_until_saturday = (5 - now_ist.weekday()) % 7  # Saturday is 5
        if days_until_saturday == 0 and now_ist.weekday() == 5:
            # Today is Saturday
            saturday = now_ist.date()
        elif days_until_saturday == 0:
            # Today is not Saturday, get next Saturday
            saturday = (now_ist + timedelta(days=7)).date()
        else:
            saturday = (now_ist + timedelta(days=days_until_saturday)).date()

        sunday = saturday + timedelta(days=1)
        return {'type': 'weekend', 'dates': [saturday, sunday]}

    # This week
    if 'this week' in query_lower:
        today = now_ist.date()
        # Get all remaining days of this week (Monday to Sunday)
        days_until_sunday = (6 - now_ist.weekday())  # Sunday is 6
        week_dates = [today + timedelta(days=i) for i in range(days_until_sunday + 1)]
        return {'type': 'this_week', 'dates': week_dates}

    return None


def validate_event_date(event: dict, date_filter: Dict) -> bool:
    """
    Validate if event's date matches the required date filter

    Args:
        event: Event dictionary with 'start_time' field
        date_filter: Date filter dictionary from detect_date_filter()

    Returns:
        True if event matches date filter, False otherwise
    """
    if not date_filter:
        return True  # No date filtering

    try:
        from datetime import datetime

        # Parse event start time
        start_time_str = event.get('start_time', '')
        if not start_time_str:
            return False

        # Handle format: "2025-12-12 18:00 IST"
        if ' IST' in start_time_str:
            date_part = start_time_str.split(' IST')[0].strip()
        else:
            date_part = start_time_str.strip()

        # Try parsing with different formats
        event_date = None
        for fmt in ['%Y-%m-%d %H:%M', '%Y-%m-%d', '%d-%m-%Y %H:%M', '%d-%m-%Y']:
            try:
                event_date = datetime.strptime(date_part, fmt).date()
                break
            except ValueError:
                continue

        if not event_date:
            return False  # Couldn't parse date

        # Check if event date is in the filter's date list
        return event_date in date_filter['dates']

    except Exception:
        return False  # Error parsing, exclude event

@app.on_event("startup")
async def startup_event():
    """Initialize bot on startup"""
    print("🚀 Starting Meetup Recommendation API...")
    try:
        # Initialize cache and load from disk
        cache = get_cache()
        if cache:
            cache.load_from_disk(max_age_hours=1)

        # Initialize session manager
        session_mgr = get_session_manager()

        # Initialize bot
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

        # Start background jobs for similar events computation
        bg_jobs = get_background_jobs()
        if bg_jobs:
            bg_jobs.start_similar_events_job(interval_hours=6)

    except Exception as e:
        print(f"⚠️  Warning: Bot initialization failed: {str(e)}")
        print("⚠️  API will start but may not function properly until bot is initialized")
        # Don't fail startup - let the API start and handle errors per request

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🛑 Shutting down Meetup Recommendation API...")
    try:
        # Save cache to disk
        cache = get_cache()
        if cache:
            cache.save_to_disk()

        # Stop background jobs
        bg_jobs = get_background_jobs()
        if bg_jobs:
            bg_jobs.stop_similar_events_job()

        # Stop event sync
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

            # PRIORITY: Check for session-based intents FIRST (before general intent detection)
            # These need to be checked early because they're context-dependent

            # Check "show more" intent - ONLY if user has an active session with events
            show_more_keywords = [
                # Direct "more" requests
                "show more", "more events", "show me more", "give me more", "any more",
                # Navigation
                "next", "next events", "show next", "next one", "next few",
                "continue", "keep going", "go on", "proceed",
                # Loading more
                "load more", "see more", "view more", "display more",
                # Alternative phrasings
                "what else", "anything else", "more options", "other options",
                "additional events", "other events", "more like this", "similar ones",
                # Casual requests
                "show some more", "got more", "have more", "any others",
                "more please", "another one", "few more", "couple more",
                # Questions
                "what other", "what more", "are there more", "do you have more",
                "is there more", "got anything else", "anything more"
            ]

            # Check if this matches "show more" keywords
            matches_show_more = message_lower.strip() == "more" or any(kw in message_lower for kw in show_more_keywords)

            if matches_show_more:
                # CRITICAL: Only handle as "show more" if user has an existing session with events
                session_mgr = get_session_manager()
                if session_mgr:
                    session_info = session_mgr.get_session_info(user_id)
                    # Check if session exists AND has events stored
                    if session_info and session_info.get('total_events', 0) > 0:
                        next_events = session_mgr.get_next_events(user_id, count=3)
                        if next_events:
                            # Add similar events to each event
                            for event in next_events:
                                similar_ids_str = event.get('similar_event_ids', '[]')
                                try:
                                    similar_ids = json.loads(similar_ids_str) if isinstance(similar_ids_str, str) else similar_ids_str
                                    if similar_ids and len(similar_ids) > 0:
                                        similar_events_data = bot.chroma_manager.collection.get(ids=similar_ids[:3])
                                        if similar_events_data and similar_events_data.get('metadatas'):
                                            event['similar_events'] = similar_events_data['metadatas']
                                except Exception as e:
                                    print(f"⚠️ Failed to fetch similar events: {e}")

                            has_more = session_info.get('has_more', False)

                            await websocket.send_json({
                                "type": "response",
                                "success": True,
                                "message": f"Here are {len(next_events)} more events for you!",
                                "recommendations": next_events,
                                "total_found": len(next_events),
                                "has_more": has_more
                            })
                            continue
                        else:
                            # No more events in session
                            await websocket.send_json({
                                "type": "response",
                                "success": False,
                                "message": "I've shown you all the events I found! Would you like to search for something different? Try asking for events in another category or location.",
                                "recommendations": [],
                                "total_found": 0,
                                "has_more": False
                            })
                            continue

                # If keyword matched BUT no session with events exists, fall through to regular search
                # This allows "show more sports events" to work as a new search on first message

            # Check "suggest best" intent - ONLY if user has an active session with events
            best_keywords = [
                # Direct best requests
                "best for me", "suggest best", "recommend best", "best events",
                "top events", "best matches", "perfect match", "ideal events",
                # Recommendations
                "top recommendations", "top picks", "top choices", "best options",
                "highly recommended", "most recommended", "your best",
                # Suitability
                "best suited", "most suitable", "perfect for me", "ideal for me",
                "just right", "right for me", "made for me",
                # Quality-focused
                "highest rated", "top rated", "most popular", "best rated",
                "premium events", "quality events", "excellent events",
                # Personalization
                "most relevant", "tailored for me", "personalized", "customized",
                "based on my interests", "matching my taste",
                # Casual requests
                "what's best", "which is best", "show best", "give me the best",
                "your top", "cream of the crop", "pick of the litter",
                # Question forms - "which one"
                "which one", "which event", "which would", "which should",
                "what would be best", "what's the best", "what is best",
                # Comparison & selection
                "better option", "better choice", "better one", "most fitting",
                "pick one", "choose one", "select best", "narrow down",
                # Help me decide
                "help me choose", "help me decide", "help me pick",
                "can you suggest", "you suggest", "you recommend",
                # From these/among these
                "from these", "among these", "out of these", "of these",
                "from the list", "from above", "from those",
                # Should I go to
                "should i go", "should i attend", "should i join",
                "which to attend", "which to join",
                # Preference queries
                "prefer which", "favorite", "favourite", "standout",
                # Priority/Focus
                "prioritize", "focus on", "go for which"
            ]

            # Check if this matches "suggest best" keywords
            matches_best = any(kw in message_lower for kw in best_keywords)

            if matches_best:
                # CRITICAL: Only handle as "suggest best" if user has an existing session with events
                session_mgr = get_session_manager()
                if session_mgr:
                    session_info = session_mgr.get_session_info(user_id)
                    # Check if session exists AND has events stored
                    if session_info and session_info.get('total_events', 0) > 0:
                        best_events = session_mgr.get_best_events(user_id, count=3)
                        if best_events:
                            # Add similar events to each event
                            for event in best_events:
                                similar_ids_str = event.get('similar_event_ids', '[]')
                                try:
                                    similar_ids = json.loads(similar_ids_str) if isinstance(similar_ids_str, str) else similar_ids_str
                                    if similar_ids and len(similar_ids) > 0:
                                        similar_events_data = bot.chroma_manager.collection.get(ids=similar_ids[:3])
                                        if similar_events_data and similar_events_data.get('metadatas'):
                                            event['similar_events'] = similar_events_data['metadatas']
                                except Exception as e:
                                    print(f"⚠️ Failed to fetch similar events: {e}")

                            await websocket.send_json({
                                "type": "response",
                                "success": True,
                                "message": f"Here are the {len(best_events)} best matches for you!",
                                "recommendations": best_events,
                                "total_found": len(best_events)
                            })
                            continue
                        else:
                            # No events in session
                            await websocket.send_json({
                                "type": "response",
                                "success": False,
                                "message": "I don't have any events to suggest yet! Let me know what activities you're interested in, and I'll find the best matches for you.",
                                "recommendations": [],
                                "total_found": 0
                            })
                            continue

                # If keyword matched BUT no session with events exists, fall through to regular search
                # This allows "best sports events" to work as a new search on first message

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

                # 4. NEW: Check if message contains any activity name
                # If user mentions an activity (football, cricket, etc.), they clearly want events
                detected_activities = detect_activities_in_query(msg)
                if detected_activities:
                    return 'search_events'  # Activity name = clear event search intent

                # 5. Event search intent - positive action words + event context
                search_patterns = ["find", "search", "show me", "looking for", "want to", "interested in",
                                 "recommend", "suggest", "get me", "i need", "i want", "looking to", "looking up", "looking around", "looking out for", "hunt for", "seeking", "explore", "exploring", "browse", "browsing", "check out", "check for", "see if there are", "see if you can find",
                                 # Vague personalized requests - should trigger event search
                                 "what is for me", "what's for me", "whats for me", "for me",
                                 "what do you have", "what you got", "what you have",
                                 "anything for me", "something for me", "events for me",
                                 "what should i", "what can i", "where should i", "where can i"]
                event_words = ["event","events", "activity", "activities", "meetup", "meetups", "meet-up",  "meet-ups", "meet","meets" , "game", "games", "match", "matches", "session", "sessions", "class","classes", "workshop", "workshops",
                "jam", "jams",  "hike", "hikes", "hiking", "run", "runs", "running", "ride","rides", "cycling", "concert", "concerts", "festival", "festivals",
                "exhibition", "exhibitions", "conference", "conferences", "seminar", "seminars", "webinar", "webinars", "party", "parties", "social", "socials", "gathering", "gatherings", "outing", "outings", "show", "shows", "performance", "performances", "tournament", "tournaments",
                 "hackathon", "hackathons", "bootcamp", "bootcamps", "retreat", "retreats", "networking", "network", "networks",
                 "lecture", "lectures", "talk", "talks", "discussion", "discussions", "debate", "debates", "league", "leagues", "championship", "championships",
                  "fitness", "workout", "workouts", "cooking", "cookings", "art", "arts", "craft", "crafts"]

                has_action = any(pattern in msg for pattern in search_patterns)
                has_event_context = any(word in msg for word in event_words)

                # Strong event intent if has action word OR event context with positive framing
                if has_action or has_event_context:
                    return 'search_events'

                # 6. Pure greeting keywords - ONLY return greeting if query is purely greeting
                # If query contains greeting + something else, should search events
                pure_greeting_keywords = ["hi", "hello", "hey", "hola", "namaste", "good morning",
                                         "good afternoon", "good evening", "good night",
                                         "greetings", "howdy", "yo", "sup", "what's up", "whats up"]

                # Check if message is ONLY a greeting (no other meaningful words)
                words = msg.split()
                if len(words) <= 3:  # Short messages only
                    if any(greet in msg for greet in pure_greeting_keywords):
                        # Check if there are other meaningful words besides greeting
                        non_greeting_words = [w for w in words if w not in pure_greeting_keywords and len(w) > 2]
                        if len(non_greeting_words) == 0:
                            return 'greeting'  # Pure greeting like "hi", "hello", "hey there"

                # 7. Default to event search for vague queries
                # Queries like "what is for me" should search events, not greet
                return 'search_events'

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

            # Get all recommendations
            all_events = recommendations_result.get("recommendations", [])

            # NEW: Detect activities in query and track search history
            detected_activities = detect_activities_in_query(query)

            # NEW: Filter events by activity if specific activity was detected
            if detected_activities:
                # Map to database activity types
                required_activity_types = map_activity_to_db_type(detected_activities)

                if required_activity_types:
                    # Filter out events that don't match the required activities
                    filtered_events = []
                    rejected_count = 0

                    for event in all_events:
                        if validate_event_activity(event, required_activity_types):
                            filtered_events.append(event)
                        else:
                            rejected_count += 1

                    # Log filtering results
                    if rejected_count > 0:
                        print(f"🔍 Filtered out {rejected_count} events not matching {required_activity_types}")
                        print(f"✅ Kept {len(filtered_events)} events matching query intent")

                    # Update all_events with filtered list
                    all_events = filtered_events

            # NEW: Filter events by date if specific date was requested
            date_filter = detect_date_filter(query)
            if date_filter:
                date_filtered_events = []
                date_rejected_count = 0

                for event in all_events:
                    if validate_event_date(event, date_filter):
                        date_filtered_events.append(event)
                    else:
                        date_rejected_count += 1

                # Log date filtering results
                if date_rejected_count > 0:
                    print(f"📅 Filtered out {date_rejected_count} events not matching date: {date_filter['type']}")
                    print(f"✅ Kept {len(date_filtered_events)} events for {date_filter['type']}")

                # Update all_events with date-filtered list
                all_events = date_filtered_events

            # Store all events in session (top 20 for "show more" functionality)
            session_mgr = get_session_manager()
            if session_mgr and all_events:
                session_mgr.store_events(user_id, all_events[:20], query)

                # Track this search in session history (in-memory only, for current conversation)
                session = session_mgr.create_or_get_session(user_id)
                session.add_search(query, user_current_city, detected_activities)

                # REMOVED: Auto-save to ChromaDB
                # Search history should NOT be saved to ChromaDB automatically
                # Searching for "football" does NOT mean user attended football events
                # Only explicit preference saving (via API) or confirmed attendance should write to ChromaDB

            # Show only top 3 events to user initially
            events_to_show = all_events[:3]

            # Mark shown events as seen in session
            if session_mgr and events_to_show:
                session = session_mgr.create_or_get_session(user_id)
                for event in events_to_show:
                    event_id = event.get('event_id') or event.get('id')
                    if event_id:
                        session.shown_events.add(event_id)
                # Update current_index to position after shown events
                session.current_index = len(events_to_show)

            # Add similar events to each event shown
            for event in events_to_show:
                similar_ids_str = event.get('similar_event_ids', '[]')
                try:
                    similar_ids = json.loads(similar_ids_str) if isinstance(similar_ids_str, str) else similar_ids_str
                    if similar_ids and len(similar_ids) > 0:
                        similar_events_data = bot.chroma_manager.collection.get(ids=similar_ids[:3])
                        if similar_events_data and similar_events_data.get('metadatas'):
                            event['similar_events'] = similar_events_data['metadatas']
                except Exception as e:
                    print(f"⚠️ Failed to fetch similar events: {e}")

            # Generate interactive response
            # CRITICAL: Only send success message if we actually have events to show
            if events_to_show:
                message = f"Found {len(events_to_show)} event{'s' if len(events_to_show) > 1 else ''} for you!"
                success = True
                has_more = len(all_events) > 3
                total_found = len(all_events)  # Use filtered count
            else:
                # NO events after filtering - always show "no events found"
                # Don't use backend's message if recommendations array is empty
                message = "No events found matching your criteria."
                success = False
                has_more = False
                total_found = 0

            response_data = {
                "type": "response",
                "success": success,
                "message": message,
                "recommendations": events_to_show,
                "total_found": total_found,
                "has_more": has_more
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