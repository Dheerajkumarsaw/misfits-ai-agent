"""
Tools for LangGraph nodes to interact with ChromaDB, LLM, and external services
These tools are used by graph nodes to perform actions
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from openai import OpenAI


class ChromaDBTool:
    """Tool for ChromaDB vector database operations"""

    def __init__(self, chroma_manager):
        """
        Initialize ChromaDB tool

        Args:
            chroma_manager: ChromaDBManager instance from ai_agent.py
        """
        self.chroma_manager = chroma_manager

    def search_events(
        self,
        query: str,
        activities: List[str] = None,
        city: str = None,
        n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for events using semantic search

        Args:
            query: Search query text
            activities: Activity filters
            city: City filter
            n_results: Number of results to return

        Returns:
            List of event dictionaries
        """
        try:
            # Build ChromaDB where filters
            where_filters = None

            # Build filter conditions
            filter_conditions = []

            if city:
                filter_conditions.append({"city_name": {"$eq": city}})

            if activities and len(activities) > 0:
                # ChromaDB supports $in operator for multiple values
                filter_conditions.append({"activity": {"$in": activities}})

            # Combine filters with $and if multiple conditions
            if len(filter_conditions) > 1:
                where_filters = {"$and": filter_conditions}
            elif len(filter_conditions) == 1:
                where_filters = filter_conditions[0]

            # Perform semantic search
            results = self.chroma_manager.search_events(
                query=query,
                n_results=n_results,
                filters=where_filters
            )

            # Filter out past events (post-process)
            current_date = datetime.now()
            filtered_results = []

            for event in results:
                start_time_str = event.get('start_time', '')
                if not start_time_str:
                    # No start time, skip
                    continue

                try:
                    # Parse start_time format: "2025-05-10 18:30 IST"
                    if ' IST' in start_time_str:
                        date_part = start_time_str.split(' IST')[0].strip()
                    else:
                        date_part = start_time_str.strip()

                    # Try parsing with different formats
                    event_date = None
                    for fmt in ['%Y-%m-%d %H:%M', '%Y-%m-%d', '%d-%m-%Y %H:%M', '%d-%m-%Y']:
                        try:
                            event_date = datetime.strptime(date_part, fmt)
                            break
                        except ValueError:
                            continue

                    # Only include future events
                    if event_date and event_date >= current_date:
                        filtered_results.append(event)

                except Exception:
                    # If parsing fails, skip this event
                    continue

            print(f"   📅 Filtered {len(results)} → {len(filtered_results)} events (removed past events)")
            return filtered_results

        except Exception as e:
            print(f"❌ ChromaDBTool.search_events error: {e}")
            return []

    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user preferences from ChromaDB

        Args:
            user_id: User identifier

        Returns:
            User preference dict or None
        """
        try:
            return self.chroma_manager.get_user_preferences_by_user_id(user_id)
        except Exception as e:
            print(f"❌ ChromaDBTool.get_user_preferences error: {e}")
            return None

    def save_user_preferences(
        self,
        user_id: str,
        activities: List[str],
        preferred_locations: List[str],
        preferred_time: str = None,
        budget_range: str = None
    ) -> bool:
        """
        Save user preferences to ChromaDB

        Args:
            user_id: User identifier
            activities: Preferred activities
            preferred_locations: Preferred locations
            preferred_time: Preferred time (optional)
            budget_range: Budget range (optional)

        Returns:
            True if saved successfully
        """
        try:
            preference_doc = {
                "user_id": user_id,
                "activities": activities,
                "preferred_locations": preferred_locations,
                "preferred_time": preferred_time,
                "budget_range": budget_range,
                "created_at": datetime.now().isoformat(),
                "activities_summary": ", ".join(activities)
            }

            return self.chroma_manager.add_user_preferences_batch([preference_doc])

        except Exception as e:
            print(f"❌ ChromaDBTool.save_user_preferences error: {e}")
            return False


class LLMTool:
    """Tool for LLM operations (intent detection, message generation)"""

    def __init__(self, llm_client: OpenAI):
        """
        Initialize LLM tool

        Args:
            llm_client: OpenAI client instance
        """
        self.client = llm_client

    def detect_intent(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        has_events_in_session: bool
    ) -> Dict[str, Any]:
        """
        Detect user intent using LLM with conversation context

        Args:
            query: Current user query
            conversation_history: Previous messages
            has_events_in_session: Whether user has events from previous search

        Returns:
            Intent classification result
        """
        # Format conversation history
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history[-6:]  # Last 6 messages
        ])

        prompt = f"""Analyze user intent in an event recommendation chatbot.

CONVERSATION HISTORY:
{history_text}

SESSION STATE:
- User has previous results: {"Yes" if has_events_in_session else "No"}

CURRENT QUERY: "{query}"

Classify intent as ONE of:
1. "new_search" - User wants to find new/different events
2. "show_more" - User wants more events from current search (requires previous results)
3. "best_picks" - User wants recommendations of best events (requires previous results)
4. "greeting" - ONLY casual greetings like "hi", "hello", "hey" (nothing else!)
5. "gratitude" - Thanking or appreciation
6. "preference_collection" - User is providing preference information

STRICT RULES (follow exactly):
1. If user has NO previous results (previous results: No):
   - Can ONLY be: "new_search", "greeting", "gratitude", or "preference_collection"
   - Even if query says "best" or "recommend" → "new_search" (need to search first!)

2. If user HAS previous results (previous results: Yes):
   - "more", "next", "show more" → "show_more"
   - "best", "recommend", "suggest", "perfect", "ideal" → "best_picks"
   - New activity/location mentioned → "new_search"

3. Short greetings ONLY:
   - "hi", "hello", "hey" alone → "greeting"
   - Greeting + request → "new_search"

4. DEFAULT: "new_search" (NOT greeting!)

Respond with ONLY a JSON object:
{{"intent": "<intent>", "confidence": 0.0-1.0, "reasoning": "<explanation>"}}"""

        # Try LLM with retries (3 attempts) before falling back to keywords
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Make API call (NVIDIA API doesn't support response_format, but model can return JSON)
                response = self.client.chat.completions.create(
                    model="meta/llama3-8b-instruct",  # Stable, fast model (verified working)
                    messages=[
                        {"role": "system", "content": "You are a precise intent classifier. Always respond with ONLY valid JSON, no other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200,
                    timeout=30
                    # Note: response_format unsupported on NVIDIA API
                )

                # Extract content from response
                content = response.choices[0].message.content
                if content:
                    content = content.strip()
                else:
                    # Empty response - retry
                    raise ValueError("Empty response from LLM API")

                # Extract JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                else:
                    # Find JSON object
                    start = content.find('{')
                    if start != -1:
                        brace_count = 0
                        for i in range(start, len(content)):
                            if content[i] == '{':
                                brace_count += 1
                            elif content[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    content = content[start:i+1]
                                    break

                result = json.loads(content)

                # Validate result structure
                if 'intent' not in result:
                    if attempt < max_retries - 1:
                        print(f"⚠️ LLM response missing 'intent' field, retrying ({attempt + 1}/{max_retries})...")
                        continue  # Try again
                    else:
                        raise ValueError("Invalid response structure from LLM")

                # Success! Return the result
                return result

            except (json.JSONDecodeError, Exception) as e:
                error_type = type(e).__name__
                error_msg = str(e)[:200]  # First 200 chars of error

                if attempt < max_retries - 1:
                    print(f"⚠️ LLM attempt {attempt + 1} failed ({error_type}): {error_msg}")
                    print(f"   Retrying ({attempt + 2}/{max_retries})...")
                    continue  # Try again
                else:
                    # Final attempt failed, fall through to keyword fallback
                    print(f"⚠️ Final attempt failed ({error_type}): {error_msg}")
                    break

        # All LLM attempts failed - use keyword fallback
        print(f"⚠️ LLM failed after {max_retries} attempts, using keyword fallback as last resort")

        # Fallback to keyword-based detection (only as last resort)
        query_lower = query.lower().strip()

        # 1. Check for gratitude/farewell (highest priority)
        gratitude_keywords = ["thanks", "thank you", "appreciate", "bye", "goodbye", "see you"]
        if any(kw in query_lower for kw in gratitude_keywords):
            return {"intent": "gratitude", "confidence": 0.8, "reasoning": "Gratitude/farewell keyword"}

        # 2. Check for greeting (only if short and contains greeting word)
        greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening"]
        is_greeting = any(query_lower.startswith(kw) or query_lower == kw for kw in greeting_keywords)
        if is_greeting and len(query_lower.split()) <= 3:
            return {"intent": "greeting", "confidence": 0.8, "reasoning": "Greeting keyword"}

        # 3. Check for show_more intent (requires previous results)
        if has_events_in_session and any(kw in query_lower for kw in ["more", "next", "show more", "more events"]):
            return {"intent": "show_more", "confidence": 0.7, "reasoning": "Show more keyword"}

        # 4. Check for best_picks/recommendation intent
        best_keywords = ["best", "recommend", "suggest", "top", "perfect", "ideal"]
        if any(kw in query_lower for kw in best_keywords):
            if has_events_in_session:
                return {"intent": "best_picks", "confidence": 0.7, "reasoning": "Best picks keyword with existing events"}
            else:
                # User wants best events but has no results yet → treat as new search
                return {"intent": "new_search", "confidence": 0.75, "reasoning": "Recommendation request (no existing events)"}

        # 5. Check for explicit search intent keywords
        search_keywords = [
            "find", "search", "looking for", "want to", "show me", "what about",
            "events", "happening", "going on", "i love", "i enjoy", "i like",
            "any", "where can i"
        ]
        if any(kw in query_lower for kw in search_keywords):
            return {"intent": "new_search", "confidence": 0.75, "reasoning": "Search keyword found"}

        # 6. Default to new_search (safer than greeting for unknown queries)
        return {"intent": "new_search", "confidence": 0.6, "reasoning": "Default to search"}

    def detect_activities(self, query: str) -> List[str]:
        """
        Detect activity types mentioned in query using LLM

        Args:
            query: User query

        Returns:
            List of detected activities
        """
        # Activity keywords mapping (same as api_server.py)
        activity_keywords = {
            'football': ['football', 'soccer', 'futsal'],
            'cricket': ['cricket'],
            'badminton': ['badminton'],
            'basketball': ['basketball', 'hoops'],
            'volleyball': ['volleyball'],
            'pickleball': ['pickleball'],
            'bowling': ['bowling'],
            'running': ['running', 'marathon', 'jogging'],
            'cycling': ['cycling', 'bike', 'biking'],
            'yoga': ['yoga', 'fitness'],
            'hiking': ['hiking', 'trek', 'trekking'],
            'dance': ['dance', 'dancing', 'zumba'],
            'music': ['music', 'concert', 'jam session'],
            'art': ['art', 'painting', 'drawing'],
            'photography': ['photography', 'photo'],
            'writing': ['writing', 'creative writing'],
            'poetry': ['poetry', 'poem', 'open mic'],
            'boardgaming': ['board game', 'boardgame', 'tabletop'],
            'video_games': ['gaming', 'game', 'esports', 'video game'],
            'chess': ['chess'],
            'drama': ['drama', 'theater', 'theatre', 'acting'],
            'films': ['movie', 'film', 'cinema'],
            'quiz': ['trivia', 'quiz'],
            'book_club': ['book club', 'reading', 'book'],
            'community_space': ['tech', 'coding', 'hackathon', 'startup'],
            'food': ['food', 'cooking', 'culinary'],
            'travel': ['travel', 'trip', 'adventure']
        }

        query_lower = query.lower()
        detected = []

        for activity, keywords in activity_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(activity.upper())

        return detected

    def generate_response_message(
        self,
        message_type: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate conversational response using LLM (Miffy personality)

        Args:
            message_type: Type of message (event_intro, show_more, best_picks, etc.)
            context: Context dict with relevant info

        Returns:
            Generated message
        """
        prompts = {
            "event_intro": f"""You are Miffy, a friendly event discovery bot. Generate ONE short sentence (max 15 words) introducing {context.get('event_count', 0)} events.

IMPORTANT: The events are for these activities ONLY: {', '.join(context.get('activities', []))}
DO NOT mention activities that are not in this list!

Example: "Great! I found 3 exciting {context.get('activities', ['event'])[0] if context.get('activities') else 'event'} events for you!"

Return ONLY the message:""",

            "show_more": f"""You are Miffy. User is seeing {context.get('new_count', 3)} more events. Generate 1-2 sentences showing progress.

The events are for: {', '.join(context.get('activities', []))}

Example: "Here are 3 more {context.get('activities', ['event'])[0] if context.get('activities') else 'event'} events - you've seen {context.get('total_shown', 6)} out of {context.get('total_available', 12)} total!"

Return ONLY the message:""",

            "best_picks": f"""You are Miffy. You're showing {context.get('pick_count', 3)} best recommendations. Generate 1-2 confident sentences.

The events are for: {', '.join(context.get('activities', []))}

Example: "Based on your preferences, these 3 {context.get('activities', ['event'])[0] if context.get('activities') else 'event'} events are perfect matches!"

Return ONLY the message:""",

            "greeting": f"""You are Miffy, a MEETUP EVENT recommendation bot. User said: "{context.get('query', '')}". Respond warmly and offer to help find meetup events.

Example: "Hey there! I'm Miffy! 👋 I help you discover amazing meetup events like sports, music, tech, arts, and more. What are you interested in?"

Return ONLY the message:""",

            "gratitude": f"""You are Miffy. User said thanks. Respond warmly (1 sentence).

Example: "You're welcome! I'm always here to help you discover great events! 😊"

Return ONLY the message:""",

            "no_events": f"""You are Miffy, a MEETUP EVENT recommendation bot. No events found for "{context.get('query', '')}" in {context.get('city', '')}.

AVAILABLE ACTIVITIES: Football, Cricket, Badminton, Basketball, Volleyball, Pickleball, Bowling, Running, Cycling, Yoga, Hiking, Dance, Music, Art, Photography, Writing, Poetry, Board Gaming, Video Games, Chess, Drama, Films, Quiz, Book Club, Tech/Coding, Food, Travel

Generate 1-2 sentences asking user to try one of these MEETUP activities. DO NOT suggest anything outside this list (no "kids activities" or "weekend getaways").

Example: "I couldn't find events for that in {context.get('city', '')}. Try searching for football, music concerts, tech meetups, or yoga classes!"

Return ONLY the message:"""
        }

        prompt = prompts.get(message_type, "Generate a friendly response.")

        try:
            response = self.client.chat.completions.create(
                model="meta/llama3-8b-instruct",  # Stable, fast model (verified working)
                messages=[
                    {"role": "system", "content": "You are Miffy, a friendly MEETUP EVENT recommendation bot. You help users discover local meetup events like sports, music, tech, arts, and community activities. Be concise and warm. NEVER suggest non-event activities like 'kids activities' or 'weekend getaways'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150,
                timeout=30  # 30 seconds is sufficient for llama3-8b
            )

            message = response.choices[0].message.content.strip().strip('"').strip("'")

            if not message:
                print(f"⚠️ LLM API returned empty message, using fallback")
                raise ValueError("Empty message from LLM API")

            return message

        except Exception as e:
            # Silently fall back to static messages (API errors are non-critical here)
            pass
            # Fallback messages (use actual activities, don't hallucinate)
            activities_str = ', '.join(context.get('activities', ['events'])) if context.get('activities') else 'events'

            fallbacks = {
                "event_intro": f"Found {context.get('event_count', 0)} {activities_str} events for you!",
                "show_more": f"Here are {context.get('new_count', 3)} more events!",
                "best_picks": f"Here are the top {context.get('pick_count', 3)} recommendations!",
                "greeting": "Hello! I'm Miffy! 👋 How can I help you find events?",
                "gratitude": "You're welcome! Happy to help! 😊",
                "no_events": f"I couldn't find events for that in {context.get('city', 'your city')}. Try searching for football, music, tech meetups, yoga, dance, or other activities!"
            }
            return fallbacks.get(message_type, "How can I help?")


class EventFormatterTool:
    """Tool for formatting events for API responses"""

    @staticmethod
    def format_events_for_response(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format event dictionaries for API response

        Args:
            events: Raw event dictionaries from ChromaDB

        Returns:
            Formatted event dictionaries
        """
        formatted = []

        for event in events:
            formatted_event = {
                "event_id": event.get("event_id") or event.get("id", ""),
                "name": event.get("event_name", ""),
                "club_name": event.get("club_name", ""),
                "activity": event.get("activity", ""),
                "start_time": event.get("start_time", ""),
                "end_time": event.get("end_time", ""),
                "location": {
                    "venue": event.get("location_name", ""),
                    "area": event.get("area_name", ""),
                    "city": event.get("city_name", ""),
                    "full_address": event.get("location_name", "")
                },
                "price": float(event.get("ticket_price", 0)),
                "available_spots": int(event.get("available_spots", 0)),
                "registration_url": event.get("event_url", ""),
                "activity_icon_url": event.get("activity_icon_url", ""),
                "club_icon_url": event.get("club_icon_url", ""),
                "event_cover_image_url": event.get("event_cover_image_url", ""),
                "event_uuid": event.get("event_uuid", ""),
                "participants_count": int(event.get("participants_count", 0)),
                "match_score": float(event.get("match_score", 0.5))
            }

            formatted.append(formatted_event)

        return formatted

    @staticmethod
    def rank_events_by_score(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank events by match score

        Args:
            events: Event list

        Returns:
            Sorted event list (highest score first)
        """
        return sorted(
            events,
            key=lambda x: x.get("match_score", 0.5),
            reverse=True
        )
