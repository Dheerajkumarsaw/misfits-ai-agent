# Interactive Meetup Recommendation Bot with ChromaDB Integration
# Install required packages at the start
from pickle import FALSE
import subprocess
import sys
# import chromadb.utils.embedding_functions

def install_package(package):
    """Install a package if not already installed"""
    try:
        __import__(package)
        print(f"✅ {package} already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")

# Install all required packages
print("🔧 Installing required packages...")
packages = ["openai", "pandas", "chromadb", "numpy", "typing-extensions", "requests", "sentence-transformers"]
for package in packages:
    install_package(package)

print("✅ All packages installed!")

# Now import all required libraries
import pandas as pd
import json
from openai import OpenAI
from datetime import datetime
import re
from google.colab import files
import io
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
import os
import requests
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Initialize the NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# EventDetailsForAgent class to match the gRPC message structure
class EventDetailsForAgent:
    def __init__(self, event_id=None, event_name="", description="", activity="",
                 start_time=None, end_time=None, allowed_friends=0, ticket_price=0,
                 event_url="", available_spots=0, location_name="", location_url="",
                 area_name="", city_name="", club_name="", payment_terms=""):
        self.event_id = event_id
        self.event_name = event_name
        self.description = self._parse_description(description)  # Handle rich text description
        self.activity = activity
        self.start_time = self._parse_datetime(start_time) if start_time else None
        self.end_time = self._parse_datetime(end_time) if end_time else None
        self.allowed_friends = allowed_friends
        self.ticket_price = ticket_price
        self.event_url = event_url
        self.available_spots = available_spots
        self.location_name = location_name
        self.location_url = location_url
        self.area_name = area_name
        self.city_name = city_name
        self.club_name = club_name
        self.payment_terms = payment_terms

    def _parse_description(self, description):
        """Parse rich text description from JSON to plain text"""
        if not description:
            return ""

        try:
            if isinstance(description, str):
                desc_data = json.loads(description)
            else:
                desc_data = description

            plain_text = ""
            for item in desc_data:
                if 'insert' in item:
                    plain_text += item['insert']
            return plain_text
        except:
            return str(description)

    def _parse_datetime(self, time_obj):
        """Parse datetime from API response format"""
        try:
            date_part = time_obj.get('date', {})
            time_part = time_obj.get('time', {})

            dt = datetime(
                year=date_part.get('year', 2025),
                month=date_part.get('month', 1),
                day=date_part.get('day', 1),
                hour=time_part.get('hour', 0),
                minute=time_part.get('minute', 0)
            )
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return str(time_obj)

    def to_dict(self):
        return {
            'event_id': self.event_id,
            'name': self.event_name,
            'description': self.description,
            'activity': self.activity,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'allowed_friends': self.allowed_friends,
            'ticket_price': self.ticket_price,
            'registration_url': self.event_url,
            'available_spots': self.available_spots,
            'location_name': self.location_name,
            'location_url': self.location_url,
            'area_name': self.area_name,
            'city_name': self.city_name,
            'club_name': self.club_name,
            'payment_terms': self.payment_terms
        }

class ChromaDBManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB manager with SentenceTransformer embeddings
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        print(f"💾 Initializing ChromaDB at: {persist_directory}")
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding function
        self.embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"  # Larger but more accurate model
        )
        
        # Initialize ChromaDB client and collection
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "meetup_events"
        self.collection = self._initialize_collection()

        # Initialize user preferences collection
        self.user_prefs_collection_name = "user_preferences"
        self.user_prefs_collection = self._initialize_user_prefs_collection()

    def _initialize_collection(self):
        """Initialize or get the events collection with embedding function"""
        print(f"🔧 Setting up collection: {self.collection_name}")

        try:
            # Get or create the collection with embedding function
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Meetup events for recommendation system"},
                embedding_function=self.embedding_function
            )
            print(f"✅ Collection ready with {collection.count()} items")
            return collection
        except Exception as e:
            print(f"❌ Failed to initialize collection: {e}")
            raise

    def _initialize_user_prefs_collection(self):
        """Initialize or get the user preferences collection with embedding function"""
        print(f"🔧 Setting up collection: {self.user_prefs_collection_name}")
        try:
            collection = self.client.get_or_create_collection(
                name=self.user_prefs_collection_name,
                metadata={"description": "User preferences and past activities for recommendations"},
                embedding_function=self.embedding_function
            )
            print(f"✅ User preferences collection ready with {collection.count()} items")
            return collection
        except Exception as e:
            print(f"❌ Failed to initialize user preferences collection: {e}")
            raise

    def prepare_event_text(self, event_data: dict) -> str:
        """
        Convert event data to searchable text with proper URL handling

        Args:
            event_data: Dictionary containing event data

        Returns:
            Formatted text for vector embedding
        """
        fields = [
            ('Event', event_data.get('name')),
            ('Club', event_data.get('club_name')),
            ('Description', self._clean_description(event_data.get('description'))),
            ('Activity', event_data.get('activity')),
            ('Start', event_data.get('start_time')),
            ('End', event_data.get('end_time')),
            ('Venue', event_data.get('location_name')),
            ('Area', event_data.get('area_name')),
            ('City', event_data.get('city_name')),
            ('Price', f"₹{event_data.get('ticket_price')}" if event_data.get('ticket_price') is not None else None),
            ('Spots', event_data.get('available_spots')),
            ('Payment', event_data.get('payment_terms')),
            ('Event Link', event_data.get('event_url')),  # Primary event access URL
            ('Location Link', event_data.get('location_url'))
        ]

        text_parts = []
        for label, value in fields:
            if value not in [None, ""]:
                text_parts.append(f"{label}: {value}")

        return " | ".join(text_parts)

    def _clean_description(self, description):
        """Clean and format the description text"""
        if not description:
            return ""

        # If description is in rich text format (list of dicts)
        if isinstance(description, list):
            try:
                return " ".join([item.get('insert', '') for item in description])
            except:
                return str(description)
        return str(description)

    def add_events_batch(self, events: List[dict], clear_existing: bool = False) -> bool:
      """
      Add batch of events to ChromaDB with proper handling of integer IDs
      
      Args:
         events: List of event dictionaries with integer IDs
         clear_existing: Whether to clear existing data first
         
      Returns:
         True if successful, False otherwise
      """
      try:
         if clear_existing:
               try:
                  existing_items = self.collection.get()
                  ids_to_delete = existing_items.get("ids", [])
                  
                  if ids_to_delete:
                     batch_size = 100
                     for i in range(0, len(ids_to_delete), batch_size):
                           batch_ids = ids_to_delete[i:i + batch_size]
                           self.collection.delete(ids=batch_ids)
                     print(f"🗑️ Deleted {len(ids_to_delete)} old events")
                  else:
                     print("ℹ️ No existing items to delete")
                     
               except Exception as e:
                  print(f"❌ Error clearing collection: {e}")
                  return False

         if not events:
               print("ℹ️ No events to add")
               return True

         documents = []
         metadatas = []
         ids = []

         for event in events:
               if not isinstance(event, dict):
                  continue

               # Handle integer event IDs - convert to string
               event_id = str(event.get('event_id')) if event.get('event_id') is not None else str(uuid.uuid4())
               
               metadata = {
                  'event_id': event_id,
                  'name': str(event.get('event_name', '')),
                  'description': self._clean_description(event.get('description')),
                  'activity': str(event.get('activity', '')),
                  'start_time': str(event.get('start_time', '')),
                  'end_time': str(event.get('end_time', '')),
                  'ticket_price': str(event.get('ticket_price', '')),
                  'event_url': str(event.get('event_url', '')),
                  'registration_url': str(event.get('event_url', '')),
                  'available_spots': str(event.get('available_spots', '')),
                  'location_name': str(event.get('location_name', '')),
                  'location_url': str(event.get('location_url', '')),
                  'area_name': str(event.get('area_name', '')),
                  'city_name': str(event.get('city_name', '')),
                  'club_name': str(event.get('club_name', '')),
                  'payment_terms': str(event.get('payment_terms', ''))
               }

               doc_text = self.prepare_event_text(metadata)
               
               documents.append(doc_text)
               metadatas.append(metadata)
               ids.append(event_id)  # Use the converted string ID

         # Add in batches with error handling
         batch_size = 100
         for i in range(0, len(documents), batch_size):
               try:
                  print(f"📦 Adding events batch {i//batch_size + 1} with {len(documents[i:i+batch_size])} items")
                  self.collection.add(
                     documents=documents[i:i+batch_size],
                     metadatas=metadatas[i:i+batch_size],
                     ids=ids[i:i+batch_size]
                  )
                  print(f"✅ Added events batch {i//batch_size + 1}")
               except Exception as e:
                  print(f"❌ Failed to add batch {i//batch_size}: {str(e)}")
                  # Try to continue with remaining batches
                  continue

         print(f"✅ Added/updated {len(documents)} events to ChromaDB")
         return True

      except Exception as e:
         print(f"❌ Critical error in add_events_batch: {str(e)}")
         return False

    def _extract_user_id(self, pref: dict) -> str:
        """Best-effort extraction of user id from preference object"""
        for key in ["user_id", "userId", "id", "uid", "_id"]:
            if key in pref and pref[key] not in [None, ""]:
                return str(pref[key])
        return str(uuid.uuid4())

    def _stringify_value(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return ", ".join([self._stringify_value(v) for v in value if v is not None])
        if isinstance(value, dict):
            parts = []
            for k, v in value.items():
                sv = self._stringify_value(v)
                if sv:
                    parts.append(f"{k}: {sv}")
            return "; ".join(parts)
        return str(value)

    def prepare_user_pref_text(self, pref: dict) -> str:
        """Convert a user preference record (GetUserPreferencesResponse.UserPreferences) to searchable text."""
        try:
            user_id = self._extract_user_id(pref)
            user_name = pref.get("user_name") or pref.get("username") or pref.get("name") or ""
            current_city = pref.get("current_city") or pref.get("city") or ""

            # Prepare activities text from repeated UserActivityPreference
            activities = pref.get("user_activities") or []
            activities_texts = []
            if isinstance(activities, list):
                for act in activities:
                    if not isinstance(act, dict):
                        continue
                    club_id = act.get("club_id", "")
                    club_name = act.get("club_name", "")
                    activity = act.get("activity", "")
                    act_city = act.get("city", "")
                    areas = act.get("area", [])
                    if isinstance(areas, list):
                        areas_text = ", ".join([self._stringify_value(a) for a in areas if a is not None])
                    else:
                        areas_text = self._stringify_value(areas)
                    attended = act.get("meetup_attended", "")
                    segment = f"{club_name} ({club_id}) | Activity: {activity} | City: {act_city} | Areas: {areas_text} | Attended: {attended}"
                    activities_texts.append(segment)

            parts = [f"User: {user_id}"]
            if user_name:
                parts.append(f"Name: {user_name}")
            if current_city:
                parts.append(f"Current City: {current_city}")
            if activities_texts:
                parts.append("Activities: " + " || ".join(activities_texts))

            # Fallback if nothing substantial present
            if len(parts) <= 1:
                fallback_fields = []
                for k, v in pref.items():
                    if isinstance(v, (str, int, float)):
                        fallback_fields.append(f"{k}: {v}")
                if fallback_fields:
                    parts.extend(fallback_fields[:10])
            return " | ".join(parts)
        except Exception:
            return self._stringify_value(pref)

    def add_user_preferences_batch(self, preferences: List[dict], clear_existing: bool = False) -> bool:
        """Add or update a batch of user preferences into the user_preferences collection."""
        try:
            print(f"🔄 add_user_preferences_batch called with {len(preferences) if preferences else 0} items (clear_existing={clear_existing})", flush=True)
            if clear_existing:
                try:
                    existing_items = self.user_prefs_collection.get()
                    ids_to_delete = existing_items.get("ids", [])
                    if ids_to_delete:
                        batch_size = 200
                        for i in range(0, len(ids_to_delete), batch_size):
                            batch_ids = ids_to_delete[i:i + batch_size]
                            self.user_prefs_collection.delete(ids=batch_ids)
                        print(f"🗑️ Deleted {len(ids_to_delete)} old user preferences")
                    else:
                        print("ℹ️ No existing user preferences to delete")
                except Exception as e:
                    print(f"❌ Error clearing user preferences: {e}")
                    return False

            if not preferences:
                print("ℹ️ No user preferences to add")
                return True

            documents = []
            metadatas = []
            ids = []
            for pref in preferences:
                if not isinstance(pref, dict):
                    continue
                user_id = self._extract_user_id(pref)
                doc_text = self.prepare_user_pref_text(pref)

                # Flatten metadata to only primitive types
                user_name = str(pref.get("user_name") or pref.get("username") or pref.get("name") or "")
                current_city = str(pref.get("current_city") or pref.get("city") or "")
                activities = pref.get("user_activities") or []
                activities_texts = []
                if isinstance(activities, list):
                    for act in activities:
                        if not isinstance(act, dict):
                            continue
                        club_name = act.get("club_name", "")
                        activity = act.get("activity", "")
                        act_city = act.get("city", "")
                        areas = act.get("area", [])
                        if isinstance(areas, list):
                            areas_text = ", ".join([self._stringify_value(a) for a in areas if a is not None])
                        else:
                            areas_text = self._stringify_value(areas)
                        attended = act.get("meetup_attended", "")
                        activities_texts.append(f"{club_name}|{activity}|{act_city}|{areas_text}|{attended}")
                activities_summary = " ; ".join(activities_texts)

                metadata = {
                    "user_id": user_id,
                    "user_name": user_name,
                    "current_city": current_city,
                    "activities_summary": activities_summary
                }
                # Ensure all metadata values are primitives
                cleaned_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        cleaned_metadata[k] = v
                    else:
                        cleaned_metadata[k] = str(v)

                documents.append(doc_text)
                metadatas.append(cleaned_metadata)
                ids.append(user_id)

            added_count = 0
            batch_size = 200
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                try:
                    print(f"📦 Adding user prefs batch {i//batch_size + 1} with {len(batch_docs)} items", flush=True)
                    self.user_prefs_collection.add(
                        documents=batch_docs,
                        metadatas=batch_meta,
                        ids=batch_ids
                    )
                    added_count += len(batch_docs)
                    print(f"✅ Added user prefs batch {i//batch_size + 1}", flush=True)
                except Exception as e:
                    print(f"❌ Failed to add user prefs batch {i//batch_size}: {str(e)}", flush=True)
                    continue
            print(f"✅ Added/updated {added_count} user preferences to ChromaDB", flush=True)
            return added_count > 0
        except Exception as e:
            print(f"❌ Critical error in add_user_preferences_batch: {str(e)}")
            return False

    def search_events(self, query: str, n_results: int = 5, filters: dict = None) -> List[dict]:
        """
        Search for events using semantic similarity with optional filters

        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Dictionary of filters to apply

        Returns:
            List of event dictionaries with metadata
        """
        try:
            params = {
                "query_texts": [query],
                "n_results": n_results
            }
            if filters:
                where = {k: {"$eq": v} for k, v in filters.items()}
                if where:
                    params["where"] = where

            results = self.collection.query(**params)

            # Combine metadata with distances for ranking
            events = []
            if results.get('metadatas'):
                for i, metadata in enumerate(results['metadatas'][0]):
                    if metadata:
                        event = metadata.copy()
                        if results.get('distances'):
                            event['similarity_score'] = 1 - results['distances'][0][i]
                        events.append(event)

            return events

        except Exception as e:
            print(f"❌ Error searching events: {e}")
            return []

    def get_event_by_id(self, event_id: str) -> Optional[dict]:
        """
        Get a single event by its ID

        Args:
            event_id: The event ID to look up

        Returns:
            The event dictionary if found, None otherwise
        """
        try:
            results = self.collection.get(
                where={"event_id": str(event_id)},
                limit=1
            )
            return results['metadatas'][0] if results['metadatas'] else None
        except Exception as e:
            print(f"❌ Error getting event by ID: {e}")
            return None

    def update_event(self, event_id: str, update_data: dict) -> bool:
        """
        Update an existing event

        Args:
            event_id: ID of the event to update
            update_data: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing event
            existing = self.collection.get(
                where={"event_id": str(event_id)},
                limit=1
            )

            if not existing['ids']:
                print(f"⚠️ Event not found: {event_id}")
                return False

            # Merge updates with existing data
            current_metadata = existing['metadatas'][0]
            updated_metadata = {**current_metadata, **{
                k: str(v) if v is not None else ""
                for k, v in update_data.items()
            }}

            # Prepare updated document
            updated_doc = self.prepare_event_text(updated_metadata)

            # Update in ChromaDB
            self.collection.update(
                ids=existing['ids'],
                documents=[updated_doc],
                metadatas=[updated_metadata]
            )

            print(f"✅ Updated event: {event_id}")
            return True

        except Exception as e:
            print(f"❌ Error updating event: {e}")
            return False

    def delete_event(self, event_id: str) -> bool:
        """
        Delete an event by ID

        Args:
            event_id: ID of the event to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            self.collection.delete(where={"event_id": str(event_id)})
            print(f"✅ Deleted event: {event_id}")
            return True
        except Exception as e:
            print(f"❌ Error deleting event: {e}")
            return False

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_events": count,
                "persist_directory": self.persist_directory,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error getting collection stats: {e}")
            return {}

    def format_event_for_display(self, event: dict) -> str:
        """
        Format an event for nice display with proper URL handling

        Args:
            event: Event dictionary from ChromaDB

        Returns:
            Formatted string for display
        """
        if not event:
            return "No event information available"

        lines = [
            f"🎉 **{event.get('name', 'Event')}**",
            f"🏷️ **Club**: {event.get('club_name', 'N/A')}",
            f"🏆 **Activity**: {event.get('activity', 'N/A')}",
            f"📅 **When**: {event.get('start_time', 'N/A')} to {event.get('end_time', 'N/A')}",
            f"📍 **Where**: {event.get('location_name', 'N/A')}",
            f"🗺️ **Area**: {event.get('area_name', 'N/A')}, {event.get('city_name', 'N/A')}",
            f"💰 **Price**: ₹{event.get('ticket_price', 'N/A')}",
            f"🎟️ **Available Spots**: {event.get('available_spots', 'N/A')}",
            f"💳 **Payment Terms**: {event.get('payment_terms', 'N/A')}"
        ]

        # Add description if available (truncated)
        if event.get('description'):
            desc = event['description']
            if len(desc) > 200:
                desc = desc[:200] + "..."
            lines.append(f"📝 **Description**: {desc}")

        # Add the primary event URL
        if event.get('event_url'):
            lines.append(f"🔗 **Event Page**: {event['event_url']}")

        # Add location URL if available
        if event.get('location_url'):
            lines.append(f"🗺️ **Location Map**: {event['location_url']}")

        return "\n".join(lines)

    def get_user_prefs_stats(self) -> dict:
        """Get statistics about the user preferences collection"""
        try:
            count = self.user_prefs_collection.count()
            return {
                "collection_name": self.user_prefs_collection_name,
                "total_user_preferences": count,
                "persist_directory": self.persist_directory,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error getting user preferences stats: {e}")
            return {}

    def get_user_preferences_by_user_id(self, user_id: str) -> Optional[dict]:
        """Fetch a single user's preferences from the user_preferences collection by user_id."""
        try:
            if not user_id:
                return None
            res = self.user_prefs_collection.get(ids=[str(user_id)])
            if not res:
                return None
            ids = res.get("ids") or []
            metas = res.get("metadatas") or []
            docs = res.get("documents") or []
            if ids and len(ids) > 0:
                return {
                    "id": ids[0],
                    "metadata": metas[0] if metas else {},
                    "document": docs[0] if docs else ""
                }
            return None
        except Exception as e:
            print(f"❌ Error fetching user preferences for user_id={user_id}: {e}")
            return None

    def get_embedding_function_info(self) -> dict:
        try:
            ef = self.embedding_function
            model_name = getattr(ef, "model_name", None)
            return {
                "type": type(ef).__name__,
                "model_name": model_name
            }
        except Exception as e:
            return {"error": str(e)}

    def validate_user_prefs_setup(self):
        """Print diagnostics helpful for fixing user preferences initialization issues."""
        try:
            print("🔎 Validating user preferences setup...", flush=True)
            print(f"📁 Persist dir: {self.persist_directory}", flush=True)
            print(f"🗂️ Events collection: {self.collection_name}", flush=True)
            print(f"🗂️ User prefs collection: {self.user_prefs_collection_name}", flush=True)
            print(f"🧠 Embedding function: {self.get_embedding_function_info()}", flush=True)
            # Try basic calls
            try:
                cnt = self.user_prefs_collection.count()
                print(f"🔢 User prefs collection count(): {cnt}", flush=True)
            except Exception as e:
                print(f"❌ Error calling count() on user prefs: {e}", flush=True)
            try:
                _ = self.user_prefs_collection.get(limit=1)
                print("✅ user_prefs_collection.get(limit=1) succeeded", flush=True)
            except Exception as e:
                print(f"❌ Error calling get() on user prefs: {e}", flush=True)
        except Exception as e:
            print(f"❌ validate_user_prefs_setup failed: {e}", flush=True)

class EventSyncManager:
    def __init__(self, chroma_manager: ChromaDBManager):
        self.upcoming_api_url = "https://notify.misfits.net.in/api/event/ai-agent/upcoming"
        self.updated_api_url = "https://notify.misfits.net.in/api/event/ai-agent/updated"
        self.chroma_manager = chroma_manager
        self.is_running = False
        self.sync_thread = None

    def call_upcoming_events_api(self) -> List[EventDetailsForAgent]:
        """Call the upcoming events REST API"""
        try:
            print(f"🔄 Calling upcoming events API: {self.upcoming_api_url}")
        
            # Prepare empty JSON payload
            payload = {}
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Note: Using POST instead of GET since GET with body is discouraged
            response = requests.post(
                self.upcoming_api_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                events = []

                # Handle both direct list and nested 'upcoming_events' format
                events_data = data.get('upcoming_events', []) if 'upcoming_events' in data else data

                for event_data in events_data:
                    event = EventDetailsForAgent(
                        event_id=event_data.get('event_id'),
                        event_name=event_data.get('event_name', ''),
                        description=event_data.get('description', ''),
                        activity=event_data.get('activity', ''),
                        start_time=event_data.get('start_time'),
                        end_time=event_data.get('end_time'),
                        allowed_friends=event_data.get('allowed_friends', 0),
                        ticket_price=event_data.get('ticket_price', 0),
                        event_url=event_data.get('event_url', ''),
                        available_spots=event_data.get('available_spots', 0),
                        location_name=event_data.get('location_name', ''),
                        location_url=event_data.get('location_url', ''),
                        area_name=event_data.get('area_name', ''),
                        city_name=event_data.get('city_name', ''),
                        club_name=event_data.get('club_name', ''),
                        payment_terms=event_data.get('payment_terms', '')
                    )
                    events.append(event)

                print(f"✅ Successfully fetched {len(events)} upcoming events")
                return events
            else:
                print(f"❌ API call failed with status code: {response.status_code}")
                return []

        except Exception as e:
            print(f"❌ Error calling upcoming events API: {e}")
            return []

    def call_updated_events_api(self) -> List[EventDetailsForAgent]:
        """Call the updated/new events REST API"""
        try:
            print(f"🔄 Calling Updated events API: {self.updated_api_url}")
            
            # Prepare empty JSON payload
            payload = {}
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Note: Using POST instead of GET since GET with body is discouraged
            response = requests.post(
                self.updated_api_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                events = []

                # Handle both direct list and nested format
                events_data = data.get('events', []) if isinstance(data, dict) else data

                for event_data in events_data:
                    event = EventDetailsForAgent(
                        event_id=event_data.get('event_id'),
                        event_name=event_data.get('event_name', ''),
                        description=event_data.get('description', ''),
                        activity=event_data.get('activity', ''),
                        start_time=event_data.get('start_time'),
                        end_time=event_data.get('end_time'),
                        allowed_friends=event_data.get('allowed_friends', 0),
                        ticket_price=event_data.get('ticket_price', 0),
                        event_url=event_data.get('event_url', ''),
                        available_spots=event_data.get('available_spots', 0),
                        location_name=event_data.get('location_name', ''),
                        location_url=event_data.get('location_url', ''),
                        area_name=event_data.get('area_name', ''),
                        city_name=event_data.get('city_name', ''),
                        club_name=event_data.get('club_name', ''),
                        payment_terms=event_data.get('payment_terms', '')
                    )
                    events.append(event)

                print(f"✅ Successfully fetched {len(events)} updated events")
                return events
            else:
                print(f"❌ API call failed with status code: {response.status_code}")
                return []

        except Exception as e:
            print(f"❌ Error calling updated events API: {e}")
            return []

    def run_single_sync(self):
        """Run a single synchronization of events from API"""
        try:
            print("🔄 Running single event synchronization...")

            # First get upcoming events
            upcoming_events = self.call_upcoming_events_api()
            upcoming_dicts = [e.to_dict() for e in upcoming_events]

            # Then get updated events
            updated_events = self.call_updated_events_api()
            updated_dicts = [e.to_dict() for e in updated_events]

            # Combine and deduplicate events
            all_events = upcoming_dicts + updated_dicts
            unique_events = {e['event_id']: e for e in all_events}.values()

            # Add to ChromaDB
            if unique_events:
                success = self.chroma_manager.add_events_batch(unique_events, clear_existing=True)
                if success:
                    print(f"✅ Successfully synchronized {len(unique_events)} events")
                else:
                    print("❌ Failed to add events to ChromaDB" )
            else:
                print("ℹ️ No events found in API responses")

        except Exception as e:
            print(f"❌ Error during synchronization: {e}")

    def start_periodic_sync(self, interval_minutes: int = 2):
        """Start periodic synchronization of events"""
        if self.is_running:
            print("⚠️ Sync is already running")
            return

        self.is_running = True
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.sync_thread.start()
        print(f"🔄 Started periodic sync every {interval_minutes} minutes")

    def stop_periodic_sync(self):
        """Stop periodic synchronization"""
        self.is_running = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join()
        print("🛑 Stopped periodic sync")

    def _sync_loop(self, interval_minutes: int):
        """Background sync loop"""
        while self.is_running:
            self.run_single_sync()
            time.sleep(interval_minutes * 60)

class UserPreferenceSyncManager:
    def __init__(self, chroma_manager: ChromaDBManager, page_limit: int = 2000):
        self.api_url = "https://notify.misfits.net.in/api/user/preferences"
        self.chroma_manager = chroma_manager
        self.page_limit = page_limit
        self.is_running = False
        self.sync_thread = None

    def _extract_list_from_response(self, data) -> List[dict]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ["user_preferences", "preferences", "data", "users", "items", "results", "list"]:
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def fetch_user_preferences_page(self, cursor: int) -> tuple:
        try:
            print(f"🔄 Fetching user preferences with cursor={cursor} (pageLimit={self.page_limit})", flush=True)
            payload = {
                "filterOptions": {
                    "cursor": cursor,
                    "offset": 0,
                    "pageLimit": self.page_limit,
                    "searchText": ""
                }
            }
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            if response.status_code != 200:
                print(f"❌ User preferences API failed with status {response.status_code}", flush=True)
                return [], None
            data = response.json()
            prefs = self._extract_list_from_response(data)
            if not prefs:
                print("ℹ️ No user preferences returned for this page", flush=True)
                return [], None
            # Determine next_cursor as the highest numeric user id in this page
            next_cursor = None
            max_numeric_id = None
            for item in prefs:
                if not isinstance(item, dict):
                    continue
                for key in ["user_id", "userId", "id", "uid", "_id"]:
                    if key in item and item[key] not in [None, ""]:
                        try:
                            candidate = int(item[key])
                        except Exception:
                            try:
                                candidate = int(str(item[key]).strip().split("-")[-1])
                            except Exception:
                                candidate = None
                        if candidate is not None:
                            if max_numeric_id is None or candidate > max_numeric_id:
                                max_numeric_id = candidate
                        break
            next_cursor = max_numeric_id
            return prefs, next_cursor
        except Exception as e:
            print(f"❌ Error fetching user preferences: {e}")
            return [], None

    def run_full_sync(self, clear_existing: bool = False):
        try:
            print("🚀 Starting full user preferences sync...", flush=True)
            cursor = 0
            seen_cursors = set()
            first_batch = True
            total = 0
            processed_ids = set()
            while True:
                if cursor in seen_cursors:
                    print("⚠️ Cursor repetition detected; incrementing to avoid loop", flush=True)
                    cursor += 1
                seen_cursors.add(cursor)
                prefs, next_cursor = self.fetch_user_preferences_page(cursor)
                if not prefs:
                    print("✅ Completed user preferences sync (no more data)")
                    break
                print(f"📄 Retrieved {len(prefs)} user preferences; page_next_cursor={next_cursor}", flush=True)

                # Deduplicate within this run
                filtered = []
                page_ids_int = []
                for p in prefs:
                    try:
                        uid_str = self.chroma_manager._extract_user_id(p)
                        uid_int = int(uid_str)
                        page_ids_int.append(uid_int)
                        if uid_str not in processed_ids:
                            filtered.append(p)
                            processed_ids.add(uid_str)
                    except Exception:
                        continue

                success = self.chroma_manager.add_user_preferences_batch(
                    filtered,
                    clear_existing=clear_existing and first_batch
                )
                first_batch = False
                if not success:
                    print("❌ Failed to upsert user preferences for current page, continuing", flush=True)
                total += len(filtered)
                # Advance cursor using the highest user_id from this page
                if page_ids_int:
                    page_max = max(page_ids_int)
                    new_cursor = page_max
                else:
                    new_cursor = next_cursor

                if new_cursor is None:
                    print("ℹ️ No valid next cursor returned; stopping.", flush=True)
                    break

                if new_cursor <= cursor:
                    print(f"↪️ Non-advancing cursor detected ({new_cursor} <= {cursor}); incrementing by 1", flush=True)
                    new_cursor = cursor + 1

                print(f"↪️ Advancing cursor: {cursor} -> {new_cursor}", flush=True)
                cursor = new_cursor
            print(f"✅ User preferences sync finished. Total records processed: {total}", flush=True)
            try:
                cnt = self.chroma_manager.user_prefs_collection.count()
                print(f"📊 User preferences collection count now: {cnt}", flush=True)
            except Exception as e:
                print(f"⚠️ Could not read user prefs count: {e}", flush=True)
        except Exception as e:
            print(f"❌ Error during user preferences sync: {e}")

    def start_periodic_sync(self, interval_hours: int = 24, clear_existing_first_run: bool = False):
        """Start periodic synchronization, defaulting to once per day."""
        if self.is_running:
            print("⚠️ User preferences sync is already running")
            return
        self.is_running = True
        def _loop():
            first = True
            while self.is_running:
                try:
                    self.run_full_sync(clear_existing=clear_existing_first_run and first)
                except Exception as e:
                    print(f"❌ Error in periodic user preferences sync: {e}")
                first = False
                sleep_seconds = max(1, int(interval_hours * 3600))
                for _ in range(sleep_seconds):
                    if not self.is_running:
                        break
                    time.sleep(1)
        self.sync_thread = threading.Thread(target=_loop, daemon=True)
        self.sync_thread.start()
        print(f"🔄 Started user preferences periodic sync every {interval_hours} hours")

    def stop_periodic_sync(self):
        """Stop periodic synchronization"""
        self.is_running = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join()
        print("🛑 Stopped user preferences periodic sync")

class MeetupBot:
    def __init__(self):
        self.events_data = None
        self.conversation_history = []
        self.chroma_manager = ChromaDBManager()
        self.event_sync_manager = EventSyncManager(self.chroma_manager)
        self.user_pref_sync_manager = UserPreferenceSyncManager(self.chroma_manager)

    def search_events_vector(self, query: str, n_results: int = 5):
        """Search events using ChromaDB vector search"""
        try:
            events = self.chroma_manager.search_events(query, n_results)
            return events
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return []

    def search_by_category_vector(self, category: str, n_results: int = 10):
        """Search events by category using ChromaDB"""
        try:
            events = self.chroma_manager.search_events(category, n_results)
            return events
        except Exception as e:
            print(f"❌ Category search error: {e}")
            return []

    def search_by_location_vector(self, location: str, n_results: int = 10):
        """Search events by location using ChromaDB"""
        try:
            events = self.chroma_manager.search_events(location, n_results)
            return events
        except Exception as e:
            print(f"❌ Location search error: {e}")
            return []

    def format_events_response(self, events: list) -> str:
        """Format events list into a readable response"""
        if not events:
            return "Sorry, I couldn't find any events matching your request. Try different keywords or check back later!"

        response = "Here are some events that might interest you:\n\n"

        for i, event in enumerate(events, 1):
            response += f"🎉 **{event.get('name', 'Event')}**\n"
            response += f"🏷️ **Club**: {event.get('club_name', 'N/A')}\n"
            response += f"🏆 **Activity**: {event.get('activity', 'N/A')}\n"
            response += f"📅 **When**: {event.get('start_time', 'N/A')} to {event.get('end_time', 'N/A')}\n"
            response += f"📍 **Where**: {event.get('location_name', 'N/A')}\n"
            response += f"🗺️ **Area**: {event.get('area_name', 'N/A')}, {event.get('city_name', 'N/A')}\n"
            response += f"💰 **Price**: ₹{event.get('ticket_price', 'N/A')}\n"
            response += f"🎟️ **Available Spots**: {event.get('available_spots', 'N/A')}\n"
            response += f"💳 **Payment Terms**: {event.get('payment_terms', 'N/A')}\n"

            if event.get('description'):
                desc = event['description']
                if len(desc) > 200:  # Truncate long descriptions
                    desc = desc[:200] + "..."
                response += f"📝 **Description**: {desc}\n"

            if event.get('registration_url'):
                response += f"🔗 **Register Here**: {event['registration_url']}\n"
            if event.get('location_url'):
                response += f"🗺️ **Location Map**: {event['location_url']}\n"

            response += "\n" + "="*50 + "\n\n"

        return response

    def prepare_context(self, user_message):
        """Prepare context with dataset information for the AI model"""
        # Optional: detect a user_id in the message to personalize recommendations
        user_pref_context = ""
        prefs_missing = False
        def _message_has_preference_clues(text: str) -> bool:
            try:
                pattern = r"\b(prefer|like|love|interested|into|my (hobbies|interests)|i want|i would like|i enjoy)\b"
                return re.search(pattern, text, flags=re.IGNORECASE) is not None
            except Exception:
                return False
        try:
            import re
            # Heuristic: look for patterns like user_id: 123 or uid 123 or just a long number token
            id_match = re.search(r"(?:user[_ ]?id|uid|id)\D*(\d{3,})", user_message, flags=re.IGNORECASE)
            if not id_match:
                # fallback: a standalone 5+ digit number
                id_match = re.search(r"\b(\d{5,})\b", user_message)
            if id_match:
                uid = id_match.group(1)
                pref = self.chroma_manager.get_user_preferences_by_user_id(uid)
                if pref:
                    md = pref.get("metadata", {}) or {}
                    user_pref_context = (
                        "\nUser Preference Context (from user_preferences):\n"
                        f"- user_id: {pref.get('id', uid)}\n"
                        f"- user_name: {md.get('user_name', 'N/A')}\n"
                        f"- current_city: {md.get('current_city', 'N/A')}\n"
                        f"- activities_summary: {md.get('activities_summary', 'N/A')}\n"
                    )
                else:
                    user_pref_context = (
                        "\nUser Preference Context: NOT FOUND for user_id " + uid + "\n"
                    )
                    prefs_missing = True
        except Exception:
            pass

        # Decide whether to search events now or ask for preferences first
        relevant_events = []
        if not prefs_missing or _message_has_preference_clues(user_message):
            relevant_events = self.search_events_vector(user_message, n_results=10)

        # Convert relevant events to string format for context
        events_context = "Relevant Events Data (from vector search):\n"
        if relevant_events:
            for event in relevant_events:
                events_context += f"🎯 Event: {event.get('name', 'N/A')}\n"
                events_context += f"  Club: {event.get('club_name', 'N/A')}\n"
                events_context += f"  Activity: {event.get('activity', 'N/A')}\n"
                events_context += f"  When: {event.get('start_time', 'N/A')} to {event.get('end_time', 'N/A')}\n"
                events_context += f"  Where: {event.get('location_name', 'N/A')} ({event.get('area_name', 'N/A')}, {event.get('city_name', 'N/A')})\n"
                events_context += f"  Price: ₹{event.get('ticket_price', 'N/A')} | Spots: {event.get('available_spots', 'N/A')}\n"
                events_context += f"  Payment: {event.get('payment_terms', 'N/A')}\n"
                events_context += f"  URL: {event.get('registration_url', 'N/A')}\n\n"
        else:
            if prefs_missing and not _message_has_preference_clues(user_message):
                events_context += "Awaiting user preferences. Ask the user about their hobbies, interests, preferred activities, city/area, budget, and timing.\n"
            else:
                events_context += "No specific events found for this query.\n"

        # Add conversation history
        history_context = "\nConversation History:\n"
        for msg in self.conversation_history[-6:]:
            history_context += f"{msg['role']}: {msg['content']}\n"

        system_prompt = f"""You are an expert, friendly meetup recommendation assistant. You have access to a vector database of events and must recommend the most relevant options based on the user's interests and constraints.

{events_context}
{history_context}
{user_pref_context}

Instructions:
1. First, interpret the user's intent (activity, date/time, budget, location, group size, preferences).
2. Return 3–7 strong, diverse options, ordered by likely relevance.
3. For each event, ALWAYS include:
   - Name and organizing club
   - Activity type
   - Date and time (local)
   - Location (venue, area, city)
   - Price and available spots
   - 1–2 line description focused on fit
   - Registration URL
4. If nothing is a direct match, provide reasonable alternatives (nearby dates/areas/categories) and tips to refine the search.
5. If the user provides a user_id, FIRST retrieve their preferences from the user_preferences collection and tailor recommendations accordingly. If no preferences are found for that user_id, explicitly state that user preferences were not found and proceed with general recommendations; invite the user to share their preferences.
6. Be concise, friendly, and helpful. Use a few tasteful emojis.
7. Never invent facts. If a field is missing, say “N/A”.
8. Prefer recent/upcoming events over past ones.

Current user message: {user_message}"""
        return system_prompt

    def get_bot_response(self, user_message):
        """Get response from the AI model with streaming"""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_message})
            # Prepare the context
            system_context = self.prepare_context(user_message)
            # Prepare messages for the API
            messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_message}
            ]
            # Make API call with streaming
            completion = client.chat.completions.create(
                model="qwen/qwq-32b",
                messages=messages,
                temperature=0.6,
                top_p=0.7,
                max_tokens=4096,
                stream=True
            )
            # Collect the streamed response
            full_response = ""
            print("Bot: ", end="", flush=True)
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print("\n")  # New line after response
            # Add bot response to history
            self.conversation_history.append({"role": "assistant", "content": full_response})
            return full_response
        except Exception as e:
            error_msg = f"❌ Error getting response: {e}"
            print(error_msg)
            return error_msg

    def start_conversation(self):
        """Start the interactive conversation loop"""
        # Check if we have data in ChromaDB
        collection_info = self.chroma_manager.get_collection_stats()
        if collection_info.get('total_events', 0) == 0:
            print("❌ No event data available. Running initial sync...")
            self.event_sync_manager.run_single_sync()
            time.sleep(2)  # Wait a moment for sync to complete

            # Check again after sync
            collection_info = self.chroma_manager.get_collection_stats()
            if collection_info.get('total_events', 0) == 0:
                print("❌ Still no events found after sync. Please check API connection.")
                return

        print("🤖 Meetup Bot is ready! Type 'quit' to exit.\n")
        print("=" * 50)
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Bot: Goodbye! Hope you find some amazing meetups! 👋")
                    break
                if not user_input:
                    continue
                # Get and display bot response
                self.get_bot_response(user_input)
            except KeyboardInterrupt:
                print("\n\nBot: Goodbye! Hope you find some amazing meetups! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def start_event_sync(self, interval_minutes: int = 2):
        """Start periodic event synchronization with API"""
        print(f"🔄 Starting event synchronization every {interval_minutes} minutes...")
        self.event_sync_manager.start_periodic_sync(interval_minutes)

    def stop_event_sync(self):
        """Stop periodic event synchronization"""
        self.event_sync_manager.stop_periodic_sync()

    def sync_events_once(self):
        """Run a single event synchronization cycle"""
        self.event_sync_manager.run_single_sync()

    def sync_user_preferences_once(self, clear_existing: bool = False):
        """Run a full user preferences synchronization cycle using cursor pagination"""
        self.user_pref_sync_manager.run_full_sync(clear_existing=clear_existing)

    def start_user_preferences_daily_sync(self, clear_existing_first_run: bool = False):
        """Start daily user preferences sync (every 24 hours)."""
        self.user_pref_sync_manager.start_periodic_sync(interval_hours=24, clear_existing_first_run=clear_existing_first_run)

    def stop_user_preferences_sync(self):
        """Stop user preferences periodic sync."""
        self.user_pref_sync_manager.stop_periodic_sync()

    def start_with_sync(self, sync_interval_minutes: int = 2):
        """Start the bot with automatic event synchronization"""
        # Start event synchronization in background
        self.start_event_sync(sync_interval_minutes)
        # Start conversation
        self.start_conversation()

# Create bot instance
bot = MeetupBot()

# Instructions for use
print("🚀 Welcome to the Interactive Meetup Recommendation Bot!")
print("=" * 60)
print("📋 Instructions:")
print("1. The bot will automatically sync events from the API")
print("2. Start chatting with the bot about events you're interested in")
print("3. The bot will recommend events based on your preferences")
print("4. Type 'quit' to exit the conversation")
print("=" * 60)

# Step 1: Check if data already exists
print("\n📁 Step 1: Checking for existing data...")
print(f"🔧 Debug: ChromaDB persist directory: {bot.chroma_manager.persist_directory}")

# Add error handling for collection info
try:
    collection_info = bot.chroma_manager.get_collection_stats()
    print(f"🔧 Debug: Collection info: {collection_info}")
    existing_events = collection_info.get('total_events', 0)
    print(f"🔧 Debug: Existing events count: {existing_events}")
except Exception as e:
    print(f"🔧 Debug: Error getting collection info: {e}")
    existing_events = 0

if existing_events > 0:
    print(f"✅ Found {existing_events} events in ChromaDB!")
else:
    print("❌ No existing events found. Running initial sync...")
    bot.sync_events_once()
    time.sleep(2)

    # Check again after sync
    try:
        collection_info = bot.chroma_manager.get_collection_stats()
        existing_events = collection_info.get('total_events', 0)
        if existing_events > 0:
            print(f"✅ Now have {existing_events} events in ChromaDB!")
        else:
            print("❌ Still no events found after sync. Please check API connection.")
    except Exception as e:
        print(f"❌ Error checking collection after sync: {e}")

# Check user preferences collection as well
print("\n📁 Step 1b: Checking user preferences data...")
try:
    user_prefs_info = bot.chroma_manager.get_user_prefs_stats()
    print(f"🔧 Debug: User preferences info: {user_prefs_info}")
    existing_user_prefs = user_prefs_info.get('total_user_preferences', 0)
    print(f"🔧 Debug: Existing user preferences count: {existing_user_prefs}")
    if existing_user_prefs == 0:
        print("ℹ️ No user preferences found. You can run: bot.sync_user_preferences_once(clear_existing=False)")
        # Run quick diagnostics to help identify setup issues
        bot.chroma_manager.validate_user_prefs_setup()
        # Try a single API page fetch to verify connectivity and payload
        try:
            print("🔌 Probing user preferences API for one page...", flush=True)
            tmp_mgr = bot.user_pref_sync_manager
            prefs, next_cursor = tmp_mgr.fetch_user_preferences_page(cursor=0)
            print(f"🧪 Probe result: received={len(prefs)} next_cursor={next_cursor}", flush=True)
            if prefs:
                print("▶️ Auto-running initial user preferences sync now...", flush=True)
                bot.sync_user_preferences_once(clear_existing=False)
        except Exception as e:
            print(f"❌ Probe call failed: {e}", flush=True)
except Exception as e:
    print(f"❌ Error checking user preferences: {e}")

if existing_events > 0:
    print("\n🎉 Great! Event data ready.")
    # Step 2: Start conversation
    print("\n💬 Step 2: Start chatting with the bot")
    print("Try asking things like:")
    print("- 'Looking for something fun to do today'")
    print("- 'I want to play football'")
    print("- 'Show me sports events'")
    print("- 'Looking to meet new people'")

    # Option to start with gRPC synchronization
    print("\n🔄 Event Synchronization Options:")
    print("1. bot.start_conversation() - Start without periodic sync")
    print("2. bot.start_with_sync() - Start with 2-minute sync")
    print("3. bot.start_with_sync(5) - Start with 5-minute sync")
    print("4. bot.sync_events_once() - Run single sync")
    print("\nCurrently using regular mode. To enable periodic sync, call:")
    print("bot.start_with_sync(2)  # for 2-minute intervals")

    bot.start_conversation()
else:
    print("❌ Please check the API connection and try again.")