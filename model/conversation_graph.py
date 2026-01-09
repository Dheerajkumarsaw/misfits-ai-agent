"""
LangGraph Conversation Flow for Meetup Recommendation Bot
Defines the state machine with nodes and conditional routing
"""

from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import datetime
import os

from graph_state import (
    ConversationState,
    update_state_with_message,
    reset_event_state,
    add_search_to_history,
    truncate_messages
)
from graph_tools import ChromaDBTool, LLMTool, EventFormatterTool


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def diversify_events_by_club(events: list, max_per_club: int = 1) -> list:
    """
    Diversify events to avoid showing too many from the same club/venue

    Args:
        events: List of event dictionaries
        max_per_club: Maximum events to show from same club (default: 1)

    Returns:
        Diversified list of events
    """
    seen_clubs = {}
    diversified = []
    remaining = []

    for event in events:
        club = event.get('club_name', '')

        if not club:
            # No club name, add to diversified
            diversified.append(event)
            continue

        club_count = seen_clubs.get(club, 0)

        if club_count < max_per_club:
            # Haven't reached limit for this club
            diversified.append(event)
            seen_clubs[club] = club_count + 1
        else:
            # Already have enough from this club, save for later
            remaining.append(event)

    # If we need more events, add remaining ones
    diversified.extend(remaining)

    return diversified


# ============================================================================
# GRAPH NODES - Each node is a step in the conversation flow
# ============================================================================

def intent_detection_node(state: ConversationState, tools: dict) -> dict:
    """
    Detect user intent using LLM with conversation context

    Args:
        state: Current conversation state
        tools: Dictionary of tool instances

    Returns:
        State updates
    """
    print(f"🔍 Intent Detection Node - Query: '{state['current_query']}'")

    llm_tool: LLMTool = tools['llm']

    # Check if user has events in session
    has_events = len(state.get('full_event_list', [])) > 0

    # Detect intent with LLM
    intent_result = llm_tool.detect_intent(
        query=state['current_query'],
        conversation_history=state.get('messages', []),
        has_events_in_session=has_events
    )

    intent = intent_result.get('intent', 'new_search')
    print(f"✅ Detected intent: {intent} (confidence: {intent_result.get('confidence', 0)})")

    # Detect activities in query if it's a search intent
    detected_activities = []
    if intent == 'new_search':
        detected_activities = llm_tool.detect_activities(state['current_query'])
        print(f"🎯 Detected activities: {detected_activities}")

    return {
        "intent": intent,
        "detected_activities": detected_activities,
        "last_updated_at": datetime.now().isoformat()
    }


def check_preferences_node(state: ConversationState, tools: dict) -> dict:
    """
    Check if user has preferences, if not flag for collection

    Args:
        state: Current conversation state
        tools: Dictionary of tool instances

    Returns:
        State updates
    """
    print(f"🔍 Check Preferences Node - User: {state['user_id']}")

    chroma_tool: ChromaDBTool = tools['chroma']

    # Get user preferences
    user_prefs = chroma_tool.get_user_preferences(state['user_id'])

    if user_prefs:
        print(f"✅ Found preferences for user {state['user_id']}")
        return {
            "user_preferences": user_prefs,
            "needs_preferences": False
        }
    else:
        print(f"⚠️  No preferences found for user {state['user_id']}")
        return {
            "user_preferences": None,
            "needs_preferences": True
        }


def resolve_search_parameters_node(state: ConversationState, tools: dict) -> dict:
    """
    Resolve final search parameters from query, preferences, and context

    Priority:
    - Activities: Detected in query > Saved preferences > Empty
    - City: user_current_city (TODO: add city detection from query)

    Args:
        state: Current conversation state
        tools: Dictionary of tool instances

    Returns:
        State updates with final_activities and final_city
    """
    print(f"🔧 Resolve Search Parameters Node")

    # Determine final activities
    final_activities = state.get('detected_activities', [])

    # If no activities in query, try saved preferences
    if len(final_activities) == 0 and state.get('user_preferences'):
        user_prefs = state.get('user_preferences', {})

        # Try new format first (activities list)
        saved_activities = user_prefs.get('activities', [])

        # If not found, try old format (metadata.activities_summary as comma-separated string)
        if not saved_activities and 'metadata' in user_prefs:
            activities_str = user_prefs['metadata'].get('activities_summary', '')
            if activities_str:
                # Parse comma-separated string and convert to uppercase
                saved_activities = [act.strip().upper() for act in activities_str.split(',')]

        if saved_activities:
            final_activities = saved_activities
            print(f"   📌 Using saved preference activities: {saved_activities}")
        else:
            print(f"   ℹ️  No activities detected or saved")
    elif len(final_activities) > 0:
        print(f"   🎯 Using query activities: {final_activities}")
    else:
        print(f"   ℹ️  No activities to search for")

    # Determine final city
    # Priority: user_current_city (for now - TODO: detect city mentions in query)
    final_city = state.get('user_current_city')

    # Note: Could check preferred_locations but user_current_city takes priority
    # since it's explicitly provided in API request
    if state.get('user_preferences'):
        user_prefs = state.get('user_preferences', {})
        preferred_locations = user_prefs.get('preferred_locations', [])
        if preferred_locations:
            print(f"   📍 User has saved city preferences: {preferred_locations} (using current: {final_city})")

    print(f"   ✅ Final search parameters: activities={final_activities}, city={final_city}")

    return {
        "final_activities": final_activities,
        "final_city": final_city
    }


def event_search_node(state: ConversationState, tools: dict) -> dict:
    """
    Search for events using ChromaDB with resolved parameters

    Args:
        state: Current conversation state
        tools: Dictionary of tool instances

    Returns:
        State updates
    """
    print(f"🔍 Event Search Node - Query: '{state['current_query']}'")

    chroma_tool: ChromaDBTool = tools['chroma']

    # Use resolved search parameters from resolve_search_parameters_node
    final_activities = state.get('final_activities', [])
    final_city = state.get('final_city')

    # Clean up query for better semantic matching
    # If we have specific activities, use activity names for better matching
    search_query = state['current_query']
    if final_activities and len(final_activities) > 0:
        # Use activity names directly for better semantic similarity
        search_query = " ".join(final_activities).lower()
        print(f"   🎯 Using cleaned search query: '{search_query}' (from activities: {final_activities})")
    else:
        print(f"   🔍 Using original query: '{search_query}'")

    # Perform semantic search (get more results for diversity filtering)
    events = chroma_tool.search_events(
        query=search_query,
        activities=final_activities,
        city=final_city,
        n_results=50  # Get more results to ensure diversity
    )

    print(f"✅ Found {len(events)} events from ChromaDB")

    # Apply club diversity to avoid showing too many from same club
    diversified_events = diversify_events_by_club(events, max_per_club=1)
    print(f"🎨 Diversified to {len(diversified_events)} events (max 1 per club)")

    # Reset event state for new search
    updates = reset_event_state(state)
    updates.update({
        "full_event_list": diversified_events,
        "has_more_events": len(diversified_events) > 3
    })

    # Add search to history
    search_update = add_search_to_history(
        query=state['current_query'],
        city=state.get('user_current_city', ''),
        activities=state.get('detected_activities', [])
    )
    updates.update(search_update)

    return updates


def show_more_node(state: ConversationState, tools: dict) -> dict:
    """
    Get next batch of events from full event list

    Args:
        state: Current conversation state
        tools: Dictionary of tool instances

    Returns:
        State updates
    """
    print(f"🔍 Show More Node - Current index: {state.get('current_index', 0)}")

    current_index = state.get('current_index', 0)
    full_list = state.get('full_event_list', [])
    shown_ids = set(state.get('shown_event_ids', []))

    # Get next 3 unseen events
    unseen = []
    new_index = current_index

    for i in range(current_index, len(full_list)):
        event = full_list[i]
        event_id = event.get('event_id') or event.get('id')

        if event_id and event_id not in shown_ids:
            unseen.append(event)
            shown_ids.add(event_id)

            if len(unseen) >= 3:
                new_index = i + 1
                break

    print(f"✅ Found {len(unseen)} more events (new index: {new_index})")

    return {
        "events_to_show": unseen,
        "shown_event_ids": list(shown_ids - set(state.get('shown_event_ids', []))),  # Only new IDs
        "current_index": new_index,
        "has_more_events": new_index < len(full_list),
        "total_events_shown": state.get('total_events_shown', 0) + len(unseen)
    }


def best_picks_node(state: ConversationState, tools: dict) -> dict:
    """
    Re-rank events and return top recommendations

    Args:
        state: Current conversation state
        tools: Dictionary of tool instances

    Returns:
        State updates
    """
    print(f"🔍 Best Picks Node - Total events: {len(state.get('full_event_list', []))}")

    formatter_tool = EventFormatterTool()
    full_list = state.get('full_event_list', [])

    # Rank by match score
    ranked = formatter_tool.rank_events_by_score(full_list)

    # Get top 3
    top_picks = ranked[:3]

    # Mark as shown
    shown_ids = [event.get('event_id') or event.get('id') for event in top_picks]

    print(f"✅ Selected {len(top_picks)} best picks")

    return {
        "events_to_show": top_picks,
        "shown_event_ids": shown_ids,
        "total_events_shown": state.get('total_events_shown', 0) + len(top_picks)
    }


def response_formatter_node(state: ConversationState, tools: dict) -> dict:
    """
    Format response message based on intent and events

    Args:
        state: Current conversation state
        tools: Dictionary of tool instances

    Returns:
        State updates
    """
    print(f"🔍 Response Formatter Node - Intent: {state.get('intent')}")

    llm_tool: LLMTool = tools['llm']
    formatter_tool = EventFormatterTool()

    intent = state.get('intent', 'new_search')
    events_to_show = state.get('events_to_show', [])

    # Check if preference collection already set a message
    if state.get('needs_preferences') and state.get('response_message'):
        # Use the preference collection message (don't overwrite it)
        message = state.get('response_message')
        print(f"✅ Using preference collection message")

    # Generate message based on intent
    elif intent == 'greeting':
        message = llm_tool.generate_response_message(
            message_type="greeting",
            context={"query": state['current_query']}
        )

    elif intent == 'gratitude':
        message = llm_tool.generate_response_message(
            message_type="gratitude",
            context={"query": state['current_query']}
        )

    elif intent == 'new_search' and events_to_show:
        # Extract unique activities from events to show
        event_activities = list(set([event.get('activity', '').lower() for event in events_to_show if event.get('activity')]))

        message = llm_tool.generate_response_message(
            message_type="event_intro",
            context={
                "event_count": len(events_to_show),
                "total_available": len(state.get('full_event_list', [])),
                "query": state['current_query'],
                "activities": event_activities  # Pass actual activities
            }
        )

    elif intent == 'show_more' and events_to_show:
        # Extract unique activities from events
        event_activities = list(set([event.get('activity', '').lower() for event in events_to_show if event.get('activity')]))

        message = llm_tool.generate_response_message(
            message_type="show_more",
            context={
                "new_count": len(events_to_show),
                "total_shown": state.get('total_events_shown', 0),
                "total_available": len(state.get('full_event_list', [])),
                "activities": event_activities
            }
        )

    elif intent == 'best_picks' and events_to_show:
        # Extract unique activities from events
        event_activities = list(set([event.get('activity', '').lower() for event in events_to_show if event.get('activity')]))

        message = llm_tool.generate_response_message(
            message_type="best_picks",
            context={
                "pick_count": len(events_to_show),
                "activities": event_activities
            }
        )

    else:
        # No events found
        message = llm_tool.generate_response_message(
            message_type="no_events",
            context={
                "query": state['current_query'],
                "city": state.get('user_current_city', '')
            }
        )

    # Format events for response
    formatted_events = formatter_tool.format_events_for_response(events_to_show)

    # Add assistant message to conversation
    message_update = update_state_with_message(state, "assistant", message)

    # Truncate messages to prevent unbounded growth
    all_messages = truncate_messages(state.get('messages', []) + message_update['messages'], max_length=20)

    print(f"✅ Generated response: {message[:100]}...")

    return {
        "response_message": message,
        "events_to_show": formatted_events,
        "messages": all_messages,  # Replace with truncated
        "last_updated_at": datetime.now().isoformat()
    }


def preference_collection_node(state: ConversationState, tools: dict) -> dict:
    """
    Request user preferences

    Args:
        state: Current conversation state
        tools: Dictionary of tool instances

    Returns:
        State updates
    """
    print(f"🔍 Preference Collection Node")

    message = """I'd love to help you find meetup events! What activities are you interested in?

We have events for: Football, Cricket, Badminton, Basketball, Volleyball, Running, Cycling, Yoga, Hiking, Dance, Music, Art, Photography, Writing, Poetry, Board Gaming, Video Games, Chess, Drama, Films, Quiz, Book Club, Tech/Coding, Food, and Travel.

What would you like to explore?"""

    message_update = update_state_with_message(state, "assistant", message)

    return {
        "response_message": message,
        "events_to_show": [],
        "messages": message_update['messages'],
        "needs_preferences": True
    }


# ============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# ============================================================================

def route_after_intent(
    state: ConversationState
) -> Literal["check_preferences", "show_more", "best_picks", "format_response"]:
    """
    Route based on detected intent

    Args:
        state: Current conversation state

    Returns:
        Next node name
    """
    intent = state.get('intent', 'new_search')

    if intent == 'greeting' or intent == 'gratitude':
        return "format_response"
    elif intent == 'show_more':
        return "show_more"
    elif intent == 'best_picks':
        return "best_picks"
    else:  # new_search
        return "check_preferences"


def route_after_preferences(
    state: ConversationState
) -> Literal["resolve_search_parameters", "collect_preferences"]:
    """
    Route based on preference availability and detected activities

    Args:
        state: Current conversation state

    Returns:
        Next node name
    """
    detected_activities = state.get('detected_activities', [])
    user_has_saved_prefs = state.get('user_preferences') is not None

    # If no activities detected in query AND no saved preferences → ask for preferences
    if len(detected_activities) == 0 and not user_has_saved_prefs:
        return "collect_preferences"

    # If user has preferences or query has activities → resolve parameters then search
    return "resolve_search_parameters"


def route_after_search(
    state: ConversationState
) -> Literal["show_initial_events", "format_response"]:
    """
    Route based on search results

    Args:
        state: Current conversation state

    Returns:
        Next node name
    """
    if len(state.get('full_event_list', [])) > 0:
        return "show_initial_events"
    else:
        return "format_response"


def show_initial_events_node(state: ConversationState, tools: dict) -> dict:
    """
    Show first 3 events from search results

    Args:
        state: Current conversation state
        tools: Dictionary of tool instances

    Returns:
        State updates
    """
    print(f"🔍 Show Initial Events Node")

    full_list = state.get('full_event_list', [])
    initial_events = full_list[:3]

    # Mark as shown
    shown_ids = [event.get('event_id') or event.get('id') for event in initial_events]

    return {
        "events_to_show": initial_events,
        "shown_event_ids": shown_ids,
        "current_index": 3,
        "has_more_events": len(full_list) > 3,
        "total_events_shown": len(initial_events)
    }


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def create_conversation_graph(tools: dict, checkpointer: Optional[SqliteSaver] = None):
    """
    Create the LangGraph state machine

    Args:
        tools: Dictionary with tool instances (chroma, llm, formatter)
        checkpointer: Optional SqliteSaver for state persistence

    Returns:
        Compiled StateGraph
    """
    # Create graph with ConversationState
    graph = StateGraph(ConversationState)

    # Add nodes (each step in conversation flow)
    graph.add_node("detect_intent", lambda state: intent_detection_node(state, tools))
    graph.add_node("check_preferences", lambda state: check_preferences_node(state, tools))
    graph.add_node("collect_preferences", lambda state: preference_collection_node(state, tools))
    graph.add_node("resolve_search_parameters", lambda state: resolve_search_parameters_node(state, tools))
    graph.add_node("search_events", lambda state: event_search_node(state, tools))
    graph.add_node("show_initial_events", lambda state: show_initial_events_node(state, tools))
    graph.add_node("show_more", lambda state: show_more_node(state, tools))
    graph.add_node("best_picks", lambda state: best_picks_node(state, tools))
    graph.add_node("format_response", lambda state: response_formatter_node(state, tools))

    # Set entry point
    graph.set_entry_point("detect_intent")

    # Add conditional routing
    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "check_preferences": "check_preferences",
            "show_more": "show_more",
            "best_picks": "best_picks",
            "format_response": "format_response"
        }
    )

    graph.add_conditional_edges(
        "check_preferences",
        route_after_preferences,
        {
            "resolve_search_parameters": "resolve_search_parameters",
            "collect_preferences": "collect_preferences"
        }
    )

    # resolve_search_parameters always goes to search_events
    graph.add_edge("resolve_search_parameters", "search_events")

    graph.add_conditional_edges(
        "search_events",
        route_after_search,
        {
            "show_initial_events": "show_initial_events",
            "format_response": "format_response"
        }
    )

    # Add edges to format_response (final node before END)
    graph.add_edge("show_initial_events", "format_response")
    graph.add_edge("show_more", "format_response")
    graph.add_edge("best_picks", "format_response")
    graph.add_edge("collect_preferences", "format_response")

    # format_response ends the graph
    graph.add_edge("format_response", END)

    # Compile the graph with optional checkpointing
    if checkpointer:
        compiled_graph = graph.compile(checkpointer=checkpointer)
        print("✅ Conversation graph compiled with SQLite state persistence")
    else:
        compiled_graph = graph.compile()
        print("✅ Conversation graph compiled (no persistence)")

    return compiled_graph


def create_sqlite_checkpointer(db_path: str = "conversation_state.db") -> SqliteSaver:
    """
    Create SQLite checkpointer for state persistence

    Args:
        db_path: Path to SQLite database file

    Returns:
        SqliteSaver instance
    """
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    # Create persistent SQLite connection
    import sqlite3
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # Create checkpointer with persistent connection
    checkpointer = SqliteSaver(conn=conn)
    print(f"✅ SQLite checkpointer created: {db_path}")

    return checkpointer
