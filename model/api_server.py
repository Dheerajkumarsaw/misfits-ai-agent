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
    from ai_agent import MeetupBot, ChromaDBManager, client as llm_client
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
        llm_client = ai_agent.client
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


def generate_conversational_message(
    message_type: str,
    context: dict
) -> str:
    """
    Universal LLM-powered message generator for natural, engaging bot responses

    Args:
        message_type: Type of message to generate
            - "event_intro": Introducing found events
            - "show_more": Showing additional events
            - "best_picks": Showing best-ranked events
            - "end_of_results": No more events to show
            - "no_session": User asks for suggestions without context
            - "capabilities": Explaining what bot can do
            - "graceful_exit": User declines/rejects
            - "gratitude": Responding to thanks/appreciation
            - "greeting": Responding to user greetings
        context: Dictionary with relevant information (query, events, user_prefs, etc.)

    Returns:
        Natural, conversational message string
    """
    import json

    # Build prompt based on message type
    prompts = {
        "event_intro": f"""You are Miffy, a friendly event discovery bot. Generate an engaging introduction for events you found.

Context:
- User query: "{context.get('query', '')}"
- Number of events: {context.get('event_count', 0)}
- Total available: {context.get('total_available', context.get('event_count', 0))}
- Activity type: {context.get('activity_type', '')}

IMPORTANT: Generate ONE friendly sentence (max 15 words).
Format: [Greeting]! I found [count] [exciting/amazing] events for you!

DO NOT:
- Mention specific event names
- Use multiple sentences
- Add extra details or descriptions
- Use phrases like "I'm excited to share" or "let me tell you about"

Good examples:
- "Great! I found 3 exciting football events for you!"
- "Awesome! I found 5 amazing dance meetups for you!"
- "Perfect! I found 2 tech events for you!"

Return ONLY the message text:""",

        "show_more": f"""You are Miffy, showing more events to an engaged user.

Context:
- New events shown: {context.get('new_count', 0)}
- Total shown so far: {context.get('total_shown', 0)}
- Total available: {context.get('total_available', 0)}
- Activity: {context.get('activity', '')}

Generate an encouraging 1-2 sentence message. Mention progress and keep engagement high.
Example: "Here are 3 more games - you've seen 6 out of 12. Still plenty more to explore!"
Return ONLY the message text:""",

        "best_picks": f"""You are Miffy, presenting your top-ranked event recommendations.

Context:
- Number of picks: {context.get('pick_count', 3)}
- Based on: {context.get('criteria', 'preferences and match score')}
- Activity: {context.get('activity', '')}

Generate an confident 1-2 sentence message explaining these are the best matches.
Example: "Based on your preferences, these 3 events are perfect matches for you!"
Return ONLY the message text:""",

        "end_of_results": f"""You are Miffy, gently informing user they've seen all results.

Context:
- Total shown: {context.get('total_shown', 0)}
- Activity: {context.get('activity', '')}
- City: {context.get('city', '')}

Generate a positive, helpful 2-3 sentence message. Suggest exploring other options.
Example: "That's all 12 events I found! Want to try different dates or nearby activities?"
Return ONLY the message text:""",

        "no_session": f"""You are Miffy, guiding user who wants suggestions without context.

Context:
- User query: "{context.get('query', '')}"

Generate a friendly 1-2 sentence message asking what they're interested in.
Example: "I'd love to suggest events! First, tell me what activities you enjoy."
Return ONLY the message text:""",

        "capabilities": f"""You are Miffy, introducing yourself and your capabilities.

Context:
- User has preferences: {context.get('has_preferences', False)}
- User city: {context.get('city', '')}

Generate a friendly 3-4 sentence introduction. Explain what you can do in a conversational way.
If user has preferences, acknowledge them. Be warm and inviting.
Return ONLY the message text:""",

        "graceful_exit": f"""You are Miffy, gracefully accepting that user doesn't want events right now.

Context:
- User query: "{context.get('query', '')}"

Generate an understanding, friendly 1-2 sentence message. Keep door open for future.
Example: "No worries! I'm here whenever you're ready to explore events."
Return ONLY the message text:""",

        "gratitude": f"""You are Miffy, responding to user's thanks or appreciation.

Context:
- User said: "{context.get('query', '')}"

Generate a warm, friendly acknowledgment (1-2 sentences). Show genuine happiness to help.
Examples:
- "You're welcome! I'm always here to help you discover great events! 😊"
- "My pleasure! Let me know if you need anything else!"
- "Glad I could help! Feel free to reach out anytime!"
Return ONLY the message text:""",

        "greeting": f"""You are Miffy, responding to a user's greeting.

Context:
- User said: "{context.get('query', '')}"

Generate a warm, friendly greeting response (1-2 sentences).
- If they asked "how are you", respond warmly and ask back
- If they said "hi/hello", greet back warmly
- Always end by offering to help find events

Examples:
- "how are you" → "I'm Miffy, and I'm doing great! 😊 How about you? Ready to discover some awesome events?"
- "hi" → "Hey there! I'm Miffy! 👋 Excited to help you find amazing events today!"
- "good morning" → "Good morning! I'm Miffy! ☀️ Hope you're having a great day! What events can I help you discover?"

Return ONLY the message text:"""
    }

    prompt = prompts.get(message_type, "")
    if not prompt:
        return "I'm here to help! Let me know what you're looking for."

    try:
        # Use global LLM client from ai_agent
        # Use standard chat model (not reasoning model) for direct user-facing messages
        completion = llm_client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",  # Changed from qwen/qwq-32b (reasoning model)
            messages=[
                {"role": "system", "content": "You are Miffy, a friendly and helpful event discovery bot. Generate natural, concise, engaging messages. Respond directly without explaining your thinking process."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Higher for more creative, varied responses
            max_tokens=150
        )

        message = completion.choices[0].message.content.strip()
        # Remove quotes if LLM wrapped the response
        message = message.strip('"').strip("'")
        return message

    except Exception as e:
        print(f"⚠️ LLM message generation failed: {e}")
        # Fallback to simple templates
        fallbacks = {
            "event_intro": f"Found {context.get('event_count', 0)} events for you!",
            "show_more": f"Here are {context.get('new_count', 3)} more events!",
            "best_picks": f"Here are the best {context.get('pick_count', 3)} matches!",
            "end_of_results": "That's all the events I found! Want to try something different?",
            "no_session": "Tell me what you're interested in, and I'll find events for you!",
            "capabilities": "I'm Miffy! I help you discover events based on your interests.",
            "graceful_exit": "No problem! I'm here whenever you need help.",
            "gratitude": "You're welcome! Happy to help anytime! 😊",
            "greeting": "Hello! I'm Miffy! 👋 Excited to help you discover amazing events today!"
        }
        return fallbacks.get(message_type, "How can I help you today?")


def generate_smart_no_results_message(
    query: str,
    detected_activities: List[str],
    date_filter: Optional[Dict],
    user_city: str,
    total_found_before_date_filter: int,
    total_found_before_activity_filter: int,
    user_preferences: Optional[dict] = None,
    query_location: Optional[str] = None
) -> str:
    """
    Generate a contextual, helpful message when no events are found

    Args:
        query: Original user query
        detected_activities: Activities detected in query
        date_filter: Date filter applied (if any)
        user_city: User's current city
        total_found_before_date_filter: Events before date filtering
        total_found_before_activity_filter: Events before activity filtering
        user_preferences: User's saved preferences for smart suggestions
        query_location: Specific city mentioned in query (if different from user_city)

    Returns:
        Smart, contextual message suggesting alternatives
    """
    # Helper: Extract alternative activities from user preferences
    def get_activity_suggestions(user_prefs: Optional[dict], current_activity: str) -> List[str]:
        """Extract 2-3 alternative activities from user preferences"""
        suggestions = []
        if user_prefs and 'metadata' in user_prefs:
            activities_summary = user_prefs.get('metadata', {}).get('activities_summary', '')
            if activities_summary:
                # Try structured format first
                import re
                activity_matches = re.findall(r'\|([A-Za-z_]+)\|', activities_summary)
                if not activity_matches and ',' in activities_summary:
                    # CSV format
                    activity_matches = [act.strip() for act in activities_summary.split(',')]

                # Get up to 3 alternatives (excluding current activity)
                for act in activity_matches:
                    act_clean = act.lower().replace('_', ' ')
                    if act_clean != current_activity.lower() and len(suggestions) < 3:
                        suggestions.append(act_clean.title())
        return suggestions

    # Determine location message - Smart priority: query location > user current city
    location_msg = f"in {query_location}" if query_location and query_location.lower() != user_city.lower() else f"in {user_city}"

    # Determine which city to use for messaging (prioritize query location)
    effective_city = query_location if query_location else user_city

    # Case 1: Activity + Date filtering removed all events
    if detected_activities and date_filter and total_found_before_date_filter > 0:
        activity_name = detected_activities[0].replace('_', ' ').title()
        date_type = date_filter['type']
        return f"I found {total_found_before_date_filter} {activity_name} event{'s' if total_found_before_date_filter > 1 else ''} {location_msg}, but none are happening {date_type}. Would you like to see events on other days, or should I suggest different activities available {date_type}?"

    # Case 2: Only date filtering removed events
    elif date_filter and total_found_before_date_filter > 0:
        date_type = date_filter['type']
        return f"I found {total_found_before_date_filter} event{'s' if total_found_before_date_filter > 1 else ''} {location_msg}, but none are happening {date_type}. Would you like to see what's available on other days?"

    # Case 3: Activity found events but they don't match query activity
    elif detected_activities and total_found_before_activity_filter > total_found_before_date_filter:
        activity_name = detected_activities[0].replace('_', ' ').title()
        suggestions = get_activity_suggestions(user_preferences, activity_name)
        suggestion_text = f" Would you like to try {' or '.join(suggestions)}?" if suggestions else " Would you like me to suggest similar activities?"
        return f"I couldn't find {activity_name} events {location_msg}.{suggestion_text}"

    # Case 4: Specific activity requested but not found
    elif detected_activities:
        activity_name = detected_activities[0].replace('_', ' ').title()
        suggestions = get_activity_suggestions(user_preferences, activity_name)
        suggestion_text = f" Would you like to try {' or '.join(suggestions)}?" if suggestions else " Would you like to explore other activities?"
        return f"I couldn't find {activity_name} events {location_msg}.{suggestion_text}"

    # Case 5: Generic search with no results
    else:
        return f"I couldn't find events matching your search {location_msg}. Could you tell me what activities interest you, or would you like to see what's trending nearby?"


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

    except HTTPException:
        # Re-raise HTTP exceptions (already formatted)
        raise
    except Exception as e:
        # Log error server-side for debugging
        print(f"❌ REST API /api/recommend Error: {e}")
        import traceback
        traceback.print_exc()
        # Return generic error to user
        raise HTTPException(status_code=500, detail="Unable to process recommendation request. Please try again.")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Enhanced chat endpoint for internal application integration
    Automatically uses user context for personalized responses
    """
    try:
        bot = get_bot()

        message_lower = request.message.lower().strip()

        # SMART INTENT DETECTION - Use LLM instead of keyword matching
        # This handles any natural language variation intelligently
        print(f"🔍 Using LLM to detect intent for: '{request.message}'")
        try:
            # Quick LLM analysis to get intent type
            preliminary_analysis = bot.analyze_user_query(request.message, {})
            llm_intent = preliminary_analysis.get('intent_type', 'event_search')
            print(f"🤖 LLM detected intent: {llm_intent}")
        except Exception as e:
            print(f"⚠️ LLM intent detection failed: {e}, falling back to 'event_search'")
            llm_intent = 'event_search'

        # Map LLM intent types to our internal intents
        # LLM returns: "greeting", "gratitude", "event_search", "bot_question", "other"
        # We need: 'greeting', 'gratitude', 'question', 'search_events', 'restart', 'negative'

        # Check for special intents that LLM might classify as "other"
        if any(kw in message_lower for kw in ["restart", "reset", "start over", "new conversation"]):
            user_intent = 'restart'
        elif any(pattern in message_lower for pattern in ["don't want", "not interested", "no thanks"]):
            user_intent = 'negative'
        elif llm_intent == 'greeting':
            user_intent = 'greeting'
        elif llm_intent == 'gratitude':
            user_intent = 'gratitude'
        elif llm_intent == 'bot_question':
            user_intent = 'question'
        elif llm_intent == 'event_search':
            user_intent = 'search_events'
        else:
            # Default to search_events for unknown intents
            user_intent = 'search_events'

        print(f"✅ Final intent: {user_intent}")

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

        elif user_intent == 'gratitude':
            # User expressing thanks - warm acknowledgment using LLM
            gratitude_response = generate_conversational_message(
                message_type="gratitude",
                context={"query": request.message}
            )
            return ChatResponse(
                success=True,
                message=gratitude_response,
                events=[],
                total_found=0,
                needs_preferences=False
            )

        elif user_intent == 'greeting':
            # Casual conversation without event intent - use LLM for natural greeting
            greeting_response = generate_conversational_message(
                message_type="greeting",
                context={"query": request.message, "user_id": request.user_id}
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
        # Log error server-side for debugging
        print(f"❌ Event sync error: {e}")
        import traceback
        traceback.print_exc()
        # Return generic error to user
        raise HTTPException(status_code=500, detail="Event synchronization failed. Please try again later.")

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
        # Log error server-side for debugging
        print(f"❌ User sync error: {e}")
        import traceback
        traceback.print_exc()
        # Return generic error to user
        raise HTTPException(status_code=500, detail="User synchronization failed. Please try again later.")

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
        # Log error server-side for debugging
        print(f"❌ User sync status error: {e}")
        import traceback
        traceback.print_exc()
        # Return generic error to user
        raise HTTPException(status_code=500, detail="Unable to retrieve sync status. Please try again later.")

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
        # Log error server-side for debugging
        print(f"❌ Readiness probe error: {e}")
        import traceback
        traceback.print_exc()
        # Return generic error to user (503 = service unavailable)
        raise HTTPException(status_code=503, detail="Service is not ready. Please try again later.")

# ============================================================================
# CONTEXT-AWARE INTENT DETECTION HELPERS
# ============================================================================

def format_conversation(messages: list) -> str:
    """Format conversation history for LLM prompt"""
    if not messages:
        return "(No previous conversation)"

    formatted = []
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def detect_intent_with_context(query: str, user_id: str, bot, session_mgr) -> dict:
    """
    Use LLM to detect user intent WITH full conversation and session context

    This replaces keyword-based and stateless intent detection with a smart
    context-aware system that understands:
    - Conversation history (what user said before)
    - Session state (does user have previous results?)
    - Query semantics (what does the query mean in context?)

    Args:
        query: Current user query
        user_id: User ID for conversation lookup
        bot: MeetupBot instance (for conversation history)
        session_mgr: SessionManager instance (for session state)

    Returns:
        {
            "intent": "best_picks" | "show_more" | "new_search" | "greeting" | "gratitude",
            "reasoning": "Explanation of why this intent was chosen",
            "confidence": 0.0-1.0
        }
    """
    # Get conversation history (last 5 messages)
    conversation = bot._get_user_conversation_history(user_id) if user_id else []
    recent_conversation = conversation[-5:] if len(conversation) > 5 else conversation

    # Get session state
    session_info = session_mgr.get_session_info(user_id) if (session_mgr and user_id) else None
    has_session = session_info and session_info.get('total_events', 0) > 0
    events_count = session_info.get('total_events', 0) if session_info else 0
    shown_count = session_info.get('current_index', 0) if session_info else 0

    # ============================================================================
    # FAST PATH: Check common keywords first (skip LLM for obvious cases)
    # ============================================================================
    query_lower = query.lower().strip()
    has_events = has_session and events_count > 0

    # Best picks keywords
    best_keywords = ['best', 'top', 'recommend', 'suggest', 'favorite', 'favourite',
                    'best for me', 'top for me', 'which is best', 'help me choose',
                    'pick for me', 'which one', 'decide', 'choose for me']

    # Show more keywords
    more_keywords = ['more', 'next', 'show more', 'what else', 'any more',
                    'continue', 'load more', 'see more']

    # Greeting keywords
    greeting_keywords = ['hi', 'hello', 'hey', 'good morning', 'good evening',
                        'how are you', 'what\'s up', 'sup']

    # Gratitude keywords
    thanks_keywords = ['thanks', 'thank you', 'appreciate', 'great', 'awesome',
                      'perfect', 'amazing']

    # Check keywords for fast return
    if has_events and any(kw in query_lower for kw in best_keywords):
        print(f"✅ Fast keyword match: 'best_picks' (session has {events_count} events)")
        return {
            "intent": "best_picks",
            "reasoning": f"Keyword match + session has {events_count} events",
            "confidence": 0.9
        }
    elif has_events and any(kw in query_lower for kw in more_keywords):
        print(f"✅ Fast keyword match: 'show_more' (session has {events_count} events)")
        return {
            "intent": "show_more",
            "reasoning": f"Keyword match + session has {events_count} events",
            "confidence": 0.9
        }
    elif any(kw in query_lower for kw in greeting_keywords):
        print(f"✅ Fast keyword match: 'greeting'")
        return {"intent": "greeting", "reasoning": "Keyword match", "confidence": 0.95}
    elif any(kw in query_lower for kw in thanks_keywords):
        print(f"✅ Fast keyword match: 'gratitude'")
        return {"intent": "gratitude", "reasoning": "Keyword match", "confidence": 0.95}

    # Build context-rich prompt for LLM
    context_prompt = f"""You are analyzing a user's intent in an event recommendation chatbot.

CONVERSATION HISTORY (last 5 messages):
{format_conversation(recent_conversation)}

SESSION STATE:
- User has previous results: {"Yes" if has_session else "No"}
- Total events available: {events_count}
- Events already shown: {shown_count}
- More events available: {"Yes" if (events_count > shown_count) else "No"}

CURRENT USER QUERY: "{query}"

TASK: Determine the user's intent from these 5 options:

1. "best_picks" - User wants you to PICK/RECOMMEND the BEST event(s) from previously shown results
   Examples: "best for me", "which one is best", "top picks", "recommend one", "help me choose",
             "which should I attend", "your top pick", "most suitable", "pick for me"
   Requirements: User MUST have previous results in session

2. "show_more" - User wants to see MORE events from the same search
   Examples: "more", "show more", "next", "what else", "any more", "continue", "load more"
   Requirements: User MUST have previous results AND more events available

3. "new_search" - User wants a NEW/DIFFERENT event search
   Examples: "show me cricket events", "find football", "events in Mumbai", "dance classes",
             any query with specific activity/location/timing
   Use this when: User mentions new criteria OR has no previous results

4. "greeting" - Casual greeting or small talk
   Examples: "hi", "hello", "how are you", "hey", "good morning"

5. "gratitude" - Thanking or appreciation
   Examples: "thanks", "thank you", "appreciate it", "great", "awesome"

CRITICAL RULES:
- If session has NO results → "best_picks" and "show_more" are IMPOSSIBLE → must be "new_search" (or greeting/gratitude)
- If query mentions specific activities (cricket, dance, etc.) → likely "new_search"
- If query is generic like "best/top" AND user has results → likely "best_picks"
- If user just got results and says "more/next" → "show_more"

IMPORTANT: You MUST respond with ONLY a valid JSON object. Do NOT include any explanations, reasoning text, or markdown.
Do NOT write "Okay, let's analyze..." or any other text. ONLY output the JSON.

Example correct response:
{{"intent": "best_picks", "reasoning": "User asked 'best for me' and has 2 previous dance events in session", "confidence": 0.95}}

Your JSON response:"""

    try:
        # Call LLM with context
        response = llm_client.chat.completions.create(
            model="qwen/qwq-32b",
            messages=[{"role": "user", "content": context_prompt}],
            temperature=0.1,
            max_tokens=200
        )

        # Check if response is valid
        if not response or not response.choices or len(response.choices) == 0:
            print(f"⚠️ Invalid API response structure")
            return {
                "intent": "new_search",
                "reasoning": "Invalid API response",
                "confidence": 0.5
            }

        # Parse LLM response
        llm_response = response.choices[0].message.content
        if llm_response is None:
            llm_response = ""
        llm_response = llm_response.strip()
        print(f"🔍 DEBUG: Raw LLM response: {llm_response[:500] if llm_response else '(empty)'}")  # Show first 500 chars

        # Handle empty response
        if not llm_response:
            print(f"⚠️ Empty LLM response, defaulting to new_search")
            return {
                "intent": "new_search",
                "reasoning": "Empty LLM response",
                "confidence": 0.5
            }

        # Extract JSON from response (handle markdown code blocks and conversational text)
        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_response:
            llm_response = llm_response.split("```")[1].split("```")[0].strip()
        else:
            # Try to find JSON object in conversational response
            # Look for { ... } pattern with proper nesting
            # Find first { and try to match balanced braces
            start = llm_response.find('{')
            if start != -1:
                # Simple approach: find matching closing brace
                brace_count = 0
                end = -1
                for i in range(start, len(llm_response)):
                    if llm_response[i] == '{':
                        brace_count += 1
                    elif llm_response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break

                if end != -1:
                    extracted_json = llm_response[start:end]
                    print(f"🔍 Extracted JSON from conversational response: {extracted_json}")
                    llm_response = extracted_json

        # Try to parse JSON
        result = json.loads(llm_response)

        print(f"🤖 Context-Aware Intent Detection:")
        print(f"   - Query: '{query}'")
        print(f"   - Session: {events_count} events, {shown_count} shown")
        print(f"   - Intent: {result.get('intent')}")
        print(f"   - Reasoning: {result.get('reasoning')}")
        print(f"   - Confidence: {result.get('confidence')}")

        return result

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
        print(f"   Raw response was: {llm_response[:200] if 'llm_response' in locals() else 'N/A'}")

        # FALLBACK: Use keyword-based detection when LLM fails
        print(f"🔄 Falling back to keyword-based intent detection")
        query_lower = query.lower().strip()

        # Check if user has session with events
        has_events = has_session and events_count > 0

        # Best picks keywords
        best_keywords = ['best', 'top', 'recommend', 'suggest', 'favorite', 'favourite',
                        'best for me', 'top for me', 'which is best', 'help me choose',
                        'pick for me', 'which one', 'decide', 'choose for me']

        # Show more keywords
        more_keywords = ['more', 'next', 'show more', 'what else', 'any more',
                        'continue', 'load more', 'see more']

        # Greeting keywords
        greeting_keywords = ['hi', 'hello', 'hey', 'good morning', 'good evening',
                            'how are you', 'what\'s up', 'sup']

        # Gratitude keywords
        thanks_keywords = ['thanks', 'thank you', 'appreciate', 'great', 'awesome',
                          'perfect', 'amazing']

        # Check keywords in order of priority
        if has_events and any(kw in query_lower for kw in best_keywords):
            print(f"✅ Keyword fallback: 'best_picks' (has {events_count} events)")
            return {
                "intent": "best_picks",
                "reasoning": f"Keyword match + session has {events_count} events",
                "confidence": 0.8
            }
        elif has_events and any(kw in query_lower for kw in more_keywords):
            print(f"✅ Keyword fallback: 'show_more' (has {events_count} events)")
            return {
                "intent": "show_more",
                "reasoning": f"Keyword match + session has {events_count} events",
                "confidence": 0.8
            }
        elif any(kw in query_lower for kw in greeting_keywords):
            print(f"✅ Keyword fallback: 'greeting'")
            return {"intent": "greeting", "reasoning": "Keyword match", "confidence": 0.9}
        elif any(kw in query_lower for kw in thanks_keywords):
            print(f"✅ Keyword fallback: 'gratitude'")
            return {"intent": "gratitude", "reasoning": "Keyword match", "confidence": 0.9}
        else:
            print(f"✅ Keyword fallback: 'new_search' (default)")
            return {
                "intent": "new_search",
                "reasoning": "No keyword match, treating as new search",
                "confidence": 0.6
            }
    except Exception as e:
        print(f"⚠️ Context-aware intent detection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "intent": "new_search",
            "reasoning": f"Error in LLM detection: {str(e)}",
            "confidence": 0.5
        }


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

            # Track user query in conversation history
            if user_id and query:
                bot._add_to_user_conversation(user_id, "user", query)
                print(f"💬 Added user message to conversation: '{query}' (user_id: {user_id})")

            # ============================================================================
            # CONTEXT-AWARE INTENT DETECTION
            # ============================================================================
            # Use LLM with full conversation and session context to determine intent
            # This replaces all keyword-based and stateless detection

            session_mgr = get_session_manager()
            intent_result = detect_intent_with_context(query, user_id, bot, session_mgr)
            user_intent = intent_result.get("intent", "new_search")

            # ============================================================================
            # HANDLE INTENT-SPECIFIC ACTIONS
            # ============================================================================

            # Intent 1: BEST_PICKS - Return top events from session
            if user_intent == "best_picks":
                best_events = session_mgr.get_best_events(user_id, count=2)
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

                    # Generate best picks message
                    best_picks_msg = generate_conversational_message(
                        message_type="best_picks",
                        context={
                            "pick_count": len(best_events),
                            "criteria": "match score and preferences",
                            "activity": ""
                        }
                    )

                    await websocket.send_json({
                        "type": "response",
                        "success": True,
                        "message": best_picks_msg,
                        "recommendations": best_events,
                        "total_found": len(best_events)
                    })
                    continue
                else:
                    print(f"⚠️ best_picks intent but no events in session")

            # Intent 2: SHOW_MORE - Return next batch from session
            elif user_intent == "show_more":
                if session_mgr:
                    next_events = session_mgr.get_next_events(user_id, count=3)
                    if next_events:
                        session_info = session_mgr.get_session_info(user_id)

                        # Add similar events
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

                        show_more_msg = generate_conversational_message(
                            message_type="show_more",
                            context={
                                "new_count": len(next_events),
                                "total_shown": session_info.get('current_index', 0),
                                "total_available": session_info.get('total_events', 0),
                                "activity": ""
                            }
                        )

                        await websocket.send_json({
                            "type": "response",
                            "success": True,
                            "message": show_more_msg,
                            "recommendations": next_events,
                            "total_found": session_info.get('total_events', len(next_events)),
                            "has_more": session_info.get('has_more', False)
                        })
                        continue
                else:
                    # No more events
                    end_msg = generate_conversational_message(
                        message_type="end_of_results",
                        context={"total_shown": 0, "activity": "", "city": user_current_city}
                    )
                    await websocket.send_json({
                        "type": "response",
                        "success": False,
                        "message": end_msg,
                        "recommendations": [],
                        "total_found": 0,
                        "has_more": False
                    })
                    continue

            # Intent 3: GREETING - Casual conversation
            elif user_intent == "greeting":
                greeting_response = generate_conversational_message(
                    message_type="greeting",
                    context={"query": query, "user_id": user_id}
                )
                await websocket.send_json({
                    "type": "response",
                    "success": True,
                    "message": greeting_response,
                    "recommendations": [],
                    "total_found": 0
                })
                continue

            # Intent 4: GRATITUDE - Thank you responses
            elif user_intent == "gratitude":
                gratitude_response = generate_conversational_message(
                    message_type="gratitude",
                    context={"query": query}
                )
                await websocket.send_json({
                    "type": "response",
                    "success": True,
                    "message": gratitude_response,
                    "recommendations": [],
                    "total_found": 0
                })
                continue

            # Intent 5: NEW_SEARCH - Falls through to get_recommendations_with_json_extraction below
            # Also handles any unrecognized intents as new search

            # Build recommendation request-compatible payload
            request_data = {
                "user_id": user_id,
                "query": query,
                "user_current_city": user_current_city,
                "preferences": {}
            }
            if user_current_city:
                request_data["preferences"]["current_city"] = user_current_city

            # ✨ SIMPLIFIED FLOW: Let AI agent handle EVERYTHING
            # Bot handles: activity detection, validation, filtering, message generation
            print(f"🔍 WebSocket: Calling get_recommendations_with_json_extraction with: {request_data}")
            try:
                result = bot.get_recommendations_with_json_extraction(request_data)
                print(f"✅ Bot returned: success={result.get('success')}, events={len(result.get('recommendations', []))}")
            except Exception as e:
                print(f"❌ WebSocket: Error calling bot: {e}")
                import traceback
                traceback.print_exc()
                result = {
                    "success": False,
                    "recommendations": [],
                    "total_found": 0,
                    "message": "I'm having trouble processing that. Please try again!"
                }

            # Handle preference collection request from bot
            if result.get('needs_preferences'):
                print(f"🚨 Bot requesting user preferences")
                await websocket.send_json({
                    "type": "response",
                    "success": False,
                    "message": result.get('message', 'Please tell me what activities you enjoy.'),
                    "recommendations": [],
                    "total_found": 0,
                    "needs_preferences": True
                })
                continue

            # Get events from bot (already validated and filtered)
            all_events = result.get("recommendations", [])

            # Store events in session for "show more" functionality
            session_mgr = get_session_manager()
            if session_mgr and all_events:
                session_mgr.store_events(user_id, all_events, query)
                print(f"💾 Stored {len(all_events)} events in session for user_id: {user_id}, query: '{query}'")

            # Show top 3 events initially
            events_to_show = all_events[:3]

            # Track shown events in session
            if session_mgr and events_to_show:
                session = session_mgr.create_or_get_session(user_id)
                for event in events_to_show:
                    event_id = event.get('event_id') or event.get('id')
                    if event_id:
                        session.shown_events.add(event_id)
                session.current_index = len(events_to_show)

            # Add similar events to each event
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

            # Generate response message
            if events_to_show:
                # Success - use bot's message or generate intro
                message = result.get('message') or generate_conversational_message(
                    message_type="event_intro",
                    context={
                        "query": query,
                        "event_count": len(events_to_show),
                        "total_available": len(all_events),
                        "activity_type": ""
                    }
                )
                success = True
                has_more = len(all_events) > 3
                total_found = len(all_events)
            else:
                # No events - use bot's message
                message = result.get('message', "I couldn't find events matching your search. Try different activities or locations!")
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

            # Track bot response in conversation history
            if user_id and message:
                bot._add_to_user_conversation(user_id, "assistant", message)
                print(f"💬 Added bot response to conversation (user_id: {user_id})")

            await websocket.send_json(response_data)
    except WebSocketDisconnect:
        # Client disconnected
        return
    except Exception as e:
        # Log error server-side for debugging
        print(f"❌ WebSocket Error: {e}")
        import traceback
        traceback.print_exc()

        # Send user-friendly response (no error details exposed)
        try:
            await websocket.send_json({
                "type": "response",
                "success": False,
                "message": "I'm having trouble processing that. Please try again or rephrase your request!",
                "recommendations": [],
                "total_found": 0
            })
        except Exception:
            # Connection already closed, silently pass
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