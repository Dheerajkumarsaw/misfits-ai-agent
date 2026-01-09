"""
LangGraph State Schema for Meetup Recommendation Bot
Defines the conversation state structure for state management
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
import operator


class ConversationState(TypedDict):
    """
    State schema for the meetup recommendation conversation graph

    This state is maintained throughout the conversation and persisted
    between user interactions for seamless multi-turn conversations.
    """

    # User Information
    user_id: str
    user_current_city: str

    # Conversation History (messages append, never replace)
    messages: Annotated[List[Dict[str, str]], operator.add]

    # Current Intent & Query
    intent: str  # "new_search", "show_more", "best_picks", "greeting", "gratitude", "preference_collection"
    current_query: str
    detected_activities: List[str]  # Activities detected in current query
    final_activities: List[str]  # Resolved activities for search (query or preferences)
    final_city: str  # Resolved city for search

    # Event Management
    full_event_list: List[Dict[str, Any]]  # All events from current search
    shown_event_ids: Annotated[List[str], operator.add]  # IDs of events already shown (append-only)
    current_index: int  # Position in event list for pagination

    # User Preferences
    user_preferences: Optional[Dict[str, Any]]
    needs_preferences: bool  # Flag to trigger preference collection

    # Search Context
    search_history: Annotated[List[Dict[str, Any]], operator.add]  # Track searches (append-only)
    activities_explored: Annotated[List[str], operator.add]  # Unique activities searched (append-only)
    cities_searched: Annotated[List[str], operator.add]  # Cities searched (append-only)

    # Response Generation
    response_message: str  # The final message to send to user
    events_to_show: List[Dict[str, Any]]  # Events to show in current response
    has_more_events: bool  # Flag indicating more events available

    # Metadata
    session_created_at: str  # ISO timestamp
    last_updated_at: str  # ISO timestamp
    total_events_shown: int  # Counter for analytics

    # Error Handling
    last_error: Optional[str]
    retry_count: int


def create_initial_state(user_id: str, user_city: str, query: str) -> ConversationState:
    """
    Create initial conversation state for a new session

    Args:
        user_id: User identifier
        user_city: User's current city
        query: Initial user query

    Returns:
        Initialized ConversationState
    """
    now = datetime.now().isoformat()

    return ConversationState(
        # User info
        user_id=user_id,
        user_current_city=user_city,

        # Conversation
        messages=[{"role": "user", "content": query}],

        # Intent
        intent="new_search",  # Default to new search
        current_query=query,
        detected_activities=[],

        # Events
        full_event_list=[],
        shown_event_ids=[],
        current_index=0,

        # Preferences
        user_preferences=None,
        needs_preferences=False,

        # Search context
        search_history=[],
        activities_explored=[],
        cities_searched=[user_city] if user_city else [],

        # Response
        response_message="",
        events_to_show=[],
        has_more_events=False,

        # Metadata
        session_created_at=now,
        last_updated_at=now,
        total_events_shown=0,

        # Error handling
        last_error=None,
        retry_count=0
    )


def update_state_with_message(state: ConversationState, role: str, content: str) -> Dict[str, Any]:
    """
    Helper to update state with a new message

    Args:
        state: Current conversation state
        role: Message role ("user" or "assistant")
        content: Message content

    Returns:
        State update dict
    """
    return {
        "messages": [{"role": role, "content": content}],
        "last_updated_at": datetime.now().isoformat()
    }


def reset_event_state(state: ConversationState) -> Dict[str, Any]:
    """
    Helper to reset event-related state for a new search

    Args:
        state: Current conversation state

    Returns:
        State update dict
    """
    return {
        "full_event_list": [],
        "shown_event_ids": [],
        "current_index": 0,
        "events_to_show": [],
        "has_more_events": False,
        "last_updated_at": datetime.now().isoformat()
    }


def add_search_to_history(
    query: str,
    city: str,
    activities: List[str]
) -> Dict[str, Any]:
    """
    Helper to add a search to history

    Args:
        query: Search query
        city: City searched
        activities: Detected activities

    Returns:
        State update dict for search history
    """
    search_entry = {
        "query": query,
        "city": city,
        "activities": activities,
        "timestamp": datetime.now().isoformat()
    }

    updates = {
        "search_history": [search_entry],
        "last_updated_at": datetime.now().isoformat()
    }

    # Add unique activities and cities (using set logic in graph)
    if activities:
        updates["activities_explored"] = activities

    if city:
        updates["cities_searched"] = [city]

    return updates


# State reducer functions for custom merge logic
def merge_unique_strings(existing: List[str], new: List[str]) -> List[str]:
    """
    Merge two lists of strings, keeping only unique values

    Args:
        existing: Existing list
        new: New list to merge

    Returns:
        Combined list with unique values
    """
    return list(set(existing + new))


def truncate_messages(messages: List[Dict[str, str]], max_length: int = 20) -> List[Dict[str, str]]:
    """
    Truncate message history to prevent unbounded growth

    Args:
        messages: Message list
        max_length: Maximum number of messages to keep

    Returns:
        Truncated message list
    """
    if len(messages) > max_length:
        return messages[-max_length:]
    return messages
