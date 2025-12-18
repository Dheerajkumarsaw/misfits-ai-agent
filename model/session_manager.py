"""
Session management for multi-turn conversations
Maintains user session state for "show more" and "suggest best" features
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import threading


class SessionContext:
    """
    User session context for maintaining conversation state

    Stores:
    - Full event list fetched from initial query
    - Events already shown to user
    - Current position in event list
    - Session metadata
    """

    def __init__(self, user_id: str):
        """
        Initialize session context

        Args:
            user_id: User identifier
        """
        self.user_id = user_id
        self.full_event_list: List[dict] = []  # All fetched events (20-30)
        self.shown_events: Set[str] = set()     # Event IDs already shown
        self.current_index: int = 0             # Current position in list
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.query_context: str = ""           # Last query for context

        # NEW: Track search history and implicit preferences
        self.search_history: List[Dict] = []    # List of {query, timestamp, activities_searched}
        self.activities_explored: Set[str] = set()  # Unique activities user searched for
        self.cities_searched: Set[str] = set()      # Cities user searched in

    def update_access_time(self):
        """Update last accessed timestamp"""
        self.last_accessed = datetime.now()

    def is_expired(self, max_age_minutes: int = 30) -> bool:
        """
        Check if session has expired

        Args:
            max_age_minutes: Maximum age in minutes

        Returns:
            True if session is expired
        """
        age = datetime.now() - self.last_accessed
        return age.total_seconds() > max_age_minutes * 60

    def has_more_events(self) -> bool:
        """Check if there are more unseen events"""
        return self.current_index < len(self.full_event_list)

    def get_unseen_count(self) -> int:
        """Get count of unseen events"""
        return len(self.full_event_list) - self.current_index

    def add_search(self, query: str, city: str = None, detected_activities: List[str] = None):
        """
        Track a search query and extract implicit preferences

        Args:
            query: User's search query
            city: City being searched in
            detected_activities: Activities detected in the query
        """
        search_entry = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'activities': detected_activities or [],
            'city': city
        }

        self.search_history.append(search_entry)

        # Track unique activities
        if detected_activities:
            for activity in detected_activities:
                self.activities_explored.add(activity.lower())

        # Track cities
        if city:
            self.cities_searched.add(city)

    def get_implicit_preferences(self) -> Dict:
        """
        Extract implicit preferences from search history

        Returns:
            Dictionary with inferred preferences
        """
        return {
            'activities_explored': list(self.activities_explored),
            'cities_searched': list(self.cities_searched),
            'total_searches': len(self.search_history),
            'search_history': self.search_history[-10:]  # Last 10 searches
        }


class SessionManager:
    """
    Manages user sessions for conversation continuity

    Features:
    - Session creation and retrieval
    - Event list storage per session
    - "Show more" pagination
    - "Suggest best" re-ranking
    - Automatic session cleanup
    """

    def __init__(self):
        """Initialize session manager"""
        self.sessions: Dict[str, SessionContext] = {}
        self._lock = threading.Lock()

        # Statistics
        self.total_sessions_created = 0
        self.total_sessions_expired = 0

        print("✅ SessionManager initialized")

    def create_or_get_session(self, user_id: str) -> SessionContext:
        """
        Get existing session or create new one

        Args:
            user_id: User identifier

        Returns:
            SessionContext for the user
        """
        with self._lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = SessionContext(user_id)
                self.total_sessions_created += 1
                print(f"🆕 Created new session for user {user_id}")
            else:
                self.sessions[user_id].update_access_time()

            return self.sessions[user_id]

    def store_events(self, user_id: str, events: List[dict], query: str = ""):
        """
        Store full event list in session
        Smart handling: Detects if new query or same query to preserve shown events

        Args:
            user_id: User identifier
            events: List of event dictionaries
            query: User's query for context
        """
        session = self.create_or_get_session(user_id)

        with self._lock:
            # Detect if this is a NEW query (different search) or continuation
            is_new_query = session.query_context != query

            if is_new_query:
                # NEW QUERY: Reset everything for fresh start
                session.full_event_list = events
                session.current_index = 0
                session.shown_events.clear()
                session.query_context = query
                print(f"💾 Stored {len(events)} NEW events for user {user_id} (query changed)")
            else:
                # SAME QUERY: Update event list but preserve shown events tracking
                session.full_event_list = events
                # Don't reset current_index or shown_events
                print(f"💾 Updated {len(events)} events for user {user_id} (same query)")

    def get_next_events(self, user_id: str, count: int = 3) -> List[dict]:
        """
        Get next N unseen events from session

        Args:
            user_id: User identifier
            count: Number of events to return

        Returns:
            List of next unseen events
        """
        session = self.sessions.get(user_id)

        if not session or not session.full_event_list:
            print(f"⚠️ No session or events for user {user_id}")
            return []

        with self._lock:
            unseen = []

            # Iterate through events starting from current_index
            for i in range(session.current_index, len(session.full_event_list)):
                event = session.full_event_list[i]
                event_id = event.get('event_id') or event.get('id')

                # Skip if already shown
                if event_id and event_id in session.shown_events:
                    continue

                unseen.append(event)

                # Mark as shown
                if event_id:
                    session.shown_events.add(event_id)

                # Stop when we have enough unseen events
                if len(unseen) >= count:
                    break

            # Update index to where we stopped searching
            if unseen:
                last_unseen_index = session.full_event_list.index(unseen[-1])
                session.current_index = last_unseen_index + 1

            session.update_access_time()

        print(f"📤 Returning {len(unseen)} unseen events for user {user_id}")
        return unseen

    def get_best_events(self, user_id: str, count: int = 3) -> List[dict]:
        """
        Re-rank and get best events (prioritizes unseen, but includes shown if needed)

        Args:
            user_id: User identifier
            count: Number of events to return

        Returns:
            List of best matching events (unseen + shown if necessary)
        """
        session = self.sessions.get(user_id)

        if not session or not session.full_event_list:
            print(f"⚠️ No session or events for user {user_id}")
            return []

        with self._lock:
            # Separate unseen and shown events
            unseen = []
            shown = []

            for event in session.full_event_list:
                event_id = event.get('event_id') or event.get('id')
                if event_id and event_id not in session.shown_events:
                    unseen.append(event)
                elif event_id and event_id in session.shown_events:
                    shown.append(event)

            # If we have enough unseen events, prioritize those
            if len(unseen) >= count:
                # Sort unseen by match score
                unseen_sorted = sorted(
                    unseen,
                    key=lambda x: x.get('match_score', 0.5),
                    reverse=True
                )
                best_events = unseen_sorted[:count]

                # Mark new ones as shown
                for event in best_events:
                    event_id = event.get('event_id') or event.get('id')
                    if event_id:
                        session.shown_events.add(event_id)

                print(f"⭐ Returning {len(best_events)} best UNSEEN events for user {user_id}")
            else:
                # Not enough unseen - combine unseen + shown and re-rank ALL
                all_events = unseen + shown

                # Sort ALL by match score
                all_sorted = sorted(
                    all_events,
                    key=lambda x: x.get('match_score', 0.5),
                    reverse=True
                )

                best_events = all_sorted[:count]

                # Mark any previously unseen as shown
                for event in best_events:
                    event_id = event.get('event_id') or event.get('id')
                    if event_id and event_id not in session.shown_events:
                        session.shown_events.add(event_id)

                print(f"⭐ Returning {len(best_events)} best events (re-ranked from shown+unseen) for user {user_id}")

            session.update_access_time()

        return best_events

    def get_session_info(self, user_id: str) -> Optional[Dict]:
        """
        Get session information

        Args:
            user_id: User identifier

        Returns:
            Session info dict or None
        """
        session = self.sessions.get(user_id)

        if not session:
            return None

        return {
            "user_id": user_id,
            "total_events": len(session.full_event_list),
            "shown_count": len(session.shown_events),
            "current_index": session.current_index,
            "has_more": session.has_more_events(),
            "unseen_count": session.get_unseen_count(),
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "query_context": session.query_context
        }

    def cleanup_expired_sessions(self, max_age_minutes: int = 30):
        """
        Remove sessions older than max_age_minutes

        Args:
            max_age_minutes: Maximum session age in minutes
        """
        with self._lock:
            expired = []

            for user_id, session in self.sessions.items():
                if session.is_expired(max_age_minutes):
                    expired.append(user_id)

            # Remove expired sessions
            for user_id in expired:
                del self.sessions[user_id]
                self.total_sessions_expired += 1

            if expired:
                print(f"🧹 Cleaned up {len(expired)} expired sessions")

    def delete_session(self, user_id: str):
        """
        Delete a specific user session

        Args:
            user_id: User identifier
        """
        with self._lock:
            if user_id in self.sessions:
                del self.sessions[user_id]
                print(f"🗑️ Deleted session for user {user_id}")

    def get_stats(self) -> Dict:
        """Get session manager statistics"""
        return {
            "active_sessions": len(self.sessions),
            "total_sessions_created": self.total_sessions_created,
            "total_sessions_expired": self.total_sessions_expired,
            "max_memory_estimate_mb": len(self.sessions) * 0.5  # Rough estimate
        }

    def clear_all_sessions(self):
        """Clear all sessions"""
        with self._lock:
            count = len(self.sessions)
            self.sessions.clear()
            print(f"🗑️ Cleared all {count} sessions")

    def save_session_to_chromadb(self, user_id: str, chroma_manager) -> bool:
        """
        Save user's session history and implicit preferences to ChromaDB

        This creates/updates a user preference entry based on search history

        Args:
            user_id: User identifier
            chroma_manager: ChromaDB manager instance

        Returns:
            True if saved successfully
        """
        session = self.sessions.get(user_id)
        if not session:
            print(f"⚠️ No session found for user {user_id}")
            return False

        try:
            import json

            # Get implicit preferences from session
            implicit_prefs = session.get_implicit_preferences()

            # Check if user already has saved preferences
            existing_prefs = chroma_manager.get_user_preferences_by_user_id(user_id)

            if existing_prefs:
                # Merge with existing preferences
                existing_metadata = existing_prefs.get('metadata', {})
                existing_activities = set(existing_metadata.get('activities_explored', []))
                new_activities = set(implicit_prefs['activities_explored'])

                # Combine activities (union of old and new)
                combined_activities = list(existing_activities.union(new_activities))

                # Merge cities
                existing_cities = set(existing_metadata.get('cities_searched', []))
                new_cities = set(implicit_prefs['cities_searched'])
                combined_cities = list(existing_cities.union(new_cities))

                # Update metadata
                updated_metadata = {
                    **existing_metadata,
                    'activities_explored': combined_activities,
                    'cities_searched': combined_cities,
                    'total_searches': existing_metadata.get('total_searches', 0) + implicit_prefs['total_searches'],
                    'last_search_history': implicit_prefs['search_history'],
                    'last_updated': datetime.now().isoformat(),
                    'source': 'session_implicit_tracking'
                }

                # Create preference document
                preference_doc = {
                    'user_id': user_id,
                    'activities': combined_activities,
                    'preferred_locations': combined_cities,
                    'activities_summary': ', '.join(combined_activities),
                    'metadata': json.dumps(updated_metadata)
                }

                print(f"📝 Updating existing preferences for user {user_id}")
                print(f"   New activities explored: {new_activities - existing_activities}")

            else:
                # Create new preference entry
                metadata = {
                    'activities_explored': implicit_prefs['activities_explored'],
                    'cities_searched': implicit_prefs['cities_searched'],
                    'total_searches': implicit_prefs['total_searches'],
                    'last_search_history': implicit_prefs['search_history'],
                    'created_at': datetime.now().isoformat(),
                    'source': 'session_implicit_tracking'
                }

                preference_doc = {
                    'user_id': user_id,
                    'activities': implicit_prefs['activities_explored'],
                    'preferred_locations': list(implicit_prefs['cities_searched']),
                    'activities_summary': ', '.join(implicit_prefs['activities_explored']),
                    'metadata': json.dumps(metadata)
                }

                print(f"🆕 Creating new preferences for user {user_id}")
                print(f"   Activities explored: {implicit_prefs['activities_explored']}")

            # Save to ChromaDB
            success = chroma_manager.add_user_preferences_batch([preference_doc])

            if success:
                print(f"✅ Saved session history to ChromaDB for user {user_id}")
                return True
            else:
                print(f"❌ Failed to save session history for user {user_id}")
                return False

        except Exception as e:
            print(f"❌ Error saving session to ChromaDB: {e}")
            import traceback
            traceback.print_exc()
            return False


# Global session manager instance (to be initialized in api_server.py)
_global_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return _global_session_manager
