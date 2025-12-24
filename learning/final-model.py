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
from datetime import datetime, timezone, timedelta
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
        """Parse datetime from API response format and convert UTC to IST"""
        try:
            date_part = time_obj.get('date', {})
            time_part = time_obj.get('time', {})

            # Create UTC datetime
            dt_utc = datetime(
                year=date_part.get('year', 2025),
                month=date_part.get('month', 1),
                day=date_part.get('day', 1),
                hour=time_part.get('hour', 0),
                minute=time_part.get('minute', 0),
                tzinfo=timezone.utc
            )
            
            # Convert to IST (UTC+5:30)
            ist_timezone = timezone(timedelta(hours=5, minutes=30))
            dt_ist = dt_utc.astimezone(ist_timezone)
            
            return dt_ist.strftime("%Y-%m-%d %H:%M IST")
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
    def __init__(self, host: str = "65.0.91.158", port: int = 8000):
        """
        Initialize ChromaDB manager with SentenceTransformer embeddings
        
        Args:
            host: ChromaDB server hostname
            port: ChromaDB server port
        """
        self.host = host
        self.port = port
        print(f"🌐 Connecting to ChromaDB at: {host}:{port}")
        
        # Initialize embedding function
        self.embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"  # Larger but more accurate model
        )
        
        # Initialize ChromaDB client and collection
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection_name = "meetup_events"
        self.collection = self._initialize_collection()

        # Initialize user preferences collection
        self.user_prefs_collection_name = "user_preferences"
        self.user_prefs_collection = self._initialize_user_prefs_collection()

    def _initialize_collection(self, force_recreate: bool = False):
        """Initialize the events collection with embedding function
        
        Args:
            force_recreate: If True, delete and recreate the collection (default: False)
        """
        print(f"🔧 Setting up collection: {self.collection_name}")

        try:
            if force_recreate:
                # Check if collection exists and delete it
                try:
                    existing_collections = [col.name for col in self.client.list_collections()]
                    if self.collection_name in existing_collections:
                        print(f"🗑️ Deleting existing collection: {self.collection_name}")
                        self.client.delete_collection(name=self.collection_name)
                        print(f"✅ Existing collection deleted")
                except Exception as e:
                    print(f"⚠️ Could not delete existing collection: {e}")
                
                # Create fresh collection
                print(f"🆕 Creating fresh collection: {self.collection_name}")
                collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Meetup events for recommendation system"},
                    embedding_function=self.embedding_function
                )
                print(f"✅ Fresh collection created with 0 items")
            else:
                # Get or create the collection with embedding function (preserve existing data)
                collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Meetup events for recommendation system"},
                    embedding_function=self.embedding_function
                )
                print(f"✅ Collection ready with {collection.count()} existing items")
            
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

    def reinitialize_events_collection(self, force_clear: bool = False):
        """Reinitialize the events collection
        
        Args:
            force_clear: If True, clears all existing data (default: False)
        """
        if force_clear:
            print("🔄 Reinitializing events collection (clearing existing data)...")
            self.collection = self._initialize_collection(force_recreate=True)
            print("✅ Events collection reinitialized with fresh start")
        else:
            print("🔄 Refreshing events collection (preserving existing data)...")
            self.collection = self._initialize_collection(force_recreate=False)
            print("✅ Events collection refreshed")
        return self.collection
    
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
               
               # Handle event URL with fallback to multiple possible field names
               event_url = (
                   event.get('event_url') or 
                   event.get('registration_url') or 
                   event.get('signup_url') or 
                   event.get('booking_url') or 
                   event.get('link') or 
                   event.get('url') or ''
               )
               
               metadata = {
                  'event_id': event_id,
                  'name': str(event.get('event_name', '')),
                  'description': self._clean_description(event.get('description')),
                  'activity': str(event.get('activity', '')),
                  'start_time': str(event.get('start_time', '')),
                  'end_time': str(event.get('end_time', '')),
                  'ticket_price': str(event.get('ticket_price', '')),
                  'event_url': str(event_url),
                  'registration_url': str(event_url),
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
            # CSV fallback: use activities_summary if no structured activities present
            activities_summary_csv = str(pref.get("activities_summary", "") or "")

            parts = [f"User: {user_id}"]
            if user_name:
                parts.append(f"Name: {user_name}")
            if current_city:
                parts.append(f"Current City: {current_city}")
            if activities_texts:
                parts.append("Activities: " + " || ".join(activities_texts))
            elif activities_summary_csv:
                parts.append("Activities: " + activities_summary_csv)

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
            print(f"🔄 DEBUG: add_user_preferences_batch called with {len(preferences) if preferences else 0} items (clear_existing={clear_existing})", flush=True)
            
            # Check if user_prefs_collection is properly initialized
            if not hasattr(self, 'user_prefs_collection') or self.user_prefs_collection is None:
                print(f"❌ DEBUG: user_prefs_collection is not initialized!", flush=True)
                return False
            
            print(f"🔧 DEBUG: user_prefs_collection type: {type(self.user_prefs_collection)}", flush=True)
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
            
            print(f"🔧 DEBUG: Starting to prepare {len(preferences)} preferences for ChromaDB", flush=True)
            
            for idx, pref in enumerate(preferences):
                if not isinstance(pref, dict):
                    print(f"❌ DEBUG: Preference {idx} is not a dict: {type(pref)}", flush=True)
                    continue
                user_id = self._extract_user_id(pref)
                doc_text = self.prepare_user_pref_text(pref)
                
                if idx < 3:  # Log first few for debugging
                    print(f"🔧 DEBUG: Preference {idx} - user_id: {user_id}", flush=True)
                    print(f"🔧 DEBUG: Preference {idx} - doc_text: {doc_text[:200]}...", flush=True)

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
                # CSV fallback: use plain activities_summary if structured activities not available
                if not activities_summary:
                    csv_summary = pref.get("activities_summary")
                    if csv_summary:
                        activities_summary = str(csv_summary)

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

            print(f"🔧 DEBUG: Prepared {len(documents)} documents, {len(metadatas)} metadatas, {len(ids)} ids", flush=True)
            
            if len(documents) != len(metadatas) or len(documents) != len(ids):
                print(f"❌ DEBUG: Array length mismatch! docs:{len(documents)}, meta:{len(metadatas)}, ids:{len(ids)}", flush=True)
                return False
            
            # Show sample of what we're about to insert
            if documents:
                print(f"🔧 DEBUG: Sample document: {documents[0][:200]}...", flush=True)
                print(f"🔧 DEBUG: Sample metadata: {metadatas[0]}", flush=True)
                print(f"🔧 DEBUG: Sample id: {ids[0]}", flush=True)
            
            added_count = 0
            batch_size = 200
            total_batches = (len(documents) + batch_size - 1) // batch_size
            print(f"🔧 DEBUG: Will process {total_batches} batches of size {batch_size}", flush=True)
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_num = i//batch_size + 1
                
                try:
                    print(f"📦 DEBUG: Adding user prefs batch {batch_num}/{total_batches} with {len(batch_docs)} items", flush=True)
                    
                    # Try the actual ChromaDB add operation
                    result = self.user_prefs_collection.add(
                        documents=batch_docs,
                        metadatas=batch_meta,
                        ids=batch_ids
                    )
                    print(f"🔧 DEBUG: ChromaDB add() returned: {result}", flush=True)
                    
                    added_count += len(batch_docs)
                    print(f"✅ DEBUG: Successfully added user prefs batch {batch_num}/{total_batches}", flush=True)
                    
                except Exception as e:
                    print(f"❌ DEBUG: Failed to add user prefs batch {batch_num}: {str(e)}", flush=True)
                    print(f"❌ DEBUG: Exception type: {type(e)}", flush=True)
                    import traceback
                    print(f"❌ DEBUG: Full traceback:\n{traceback.format_exc()}", flush=True)
                    continue
                    
            print(f"✅ DEBUG: Final result - Added/updated {added_count} user preferences to ChromaDB", flush=True)
            
            # Verify the data was actually saved
            try:
                collection_count = self.user_prefs_collection.count()
                print(f"🔧 DEBUG: Collection count after insertion: {collection_count}", flush=True)
            except Exception as e:
                print(f"❌ DEBUG: Error checking collection count: {e}", flush=True)
            
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
                "host": self.host,
                "port": self.port,
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
                "host": self.host,
                "port": self.port,
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
            print(f"🌐 ChromaDB server: {self.host}:{self.port}", flush=True)
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
                
                # Debug: Check if we have any events with URLs
                if events_data and len(events_data) > 0:
                    first_event = events_data[0]
                    url_fields = ['event_url', 'registration_url', 'signup_url', 'booking_url', 'link', 'url']
                    found_urls = {field: first_event.get(field) for field in url_fields if first_event.get(field)}
                    print(f"🔧 Debug: First upcoming event URL fields found: {found_urls}")
                    if not found_urls:
                        print(f"⚠️ Warning: No URL fields found in first upcoming event. Available fields: {list(first_event.keys())}")
                else:
                    print("⚠️ Warning: No events data received from upcoming API")

                for event_data in events_data:
                    # Try multiple possible URL field names from the API
                    event_url = (
                        event_data.get('event_url') or 
                        event_data.get('registration_url') or 
                        event_data.get('signup_url') or 
                        event_data.get('booking_url') or 
                        event_data.get('link') or 
                        event_data.get('url') or ''
                    )
                    
                    event = EventDetailsForAgent(
                        event_id=event_data.get('event_id'),
                        event_name=event_data.get('event_name', ''),
                        description=event_data.get('description', ''),
                        activity=event_data.get('activity', ''),
                        start_time=event_data.get('start_time'),
                        end_time=event_data.get('end_time'),
                        allowed_friends=event_data.get('allowed_friends', 0),
                        ticket_price=event_data.get('ticket_price', 0),
                        event_url=event_url,
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
                
                # Debug: Check if we have any events with URLs
                if events_data and len(events_data) > 0:
                    first_event = events_data[0]
                    url_fields = ['event_url', 'registration_url', 'signup_url', 'booking_url', 'link', 'url']
                    found_urls = {field: first_event.get(field) for field in url_fields if first_event.get(field)}
                    print(f"🔧 Debug: First updated event URL fields found: {found_urls}")
                    if not found_urls:
                        print(f"⚠️ Warning: No URL fields found in first updated event. Available fields: {list(first_event.keys())}")
                else:
                    print("⚠️ Warning: No events data received from updated API")

                for event_data in events_data:
                    # Try multiple possible URL field names from the API
                    event_url = (
                        event_data.get('event_url') or 
                        event_data.get('registration_url') or 
                        event_data.get('signup_url') or 
                        event_data.get('booking_url') or 
                        event_data.get('link') or 
                        event_data.get('url') or ''
                    )
                    
                    event = EventDetailsForAgent(
                        event_id=event_data.get('event_id'),
                        event_name=event_data.get('event_name', ''),
                        description=event_data.get('description', ''),
                        activity=event_data.get('activity', ''),
                        start_time=event_data.get('start_time'),
                        end_time=event_data.get('end_time'),
                        allowed_friends=event_data.get('allowed_friends', 0),
                        ticket_price=event_data.get('ticket_price', 0),
                        event_url=event_url,
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

    def run_single_sync(self, full_sync: bool = False):
        """Run a single synchronization of events from API
        
        Args:
            full_sync: If True, performs full sync with upcoming+updated events (default: False)
                      If False, only syncs updated events for incremental updates
        """
        try:
            if full_sync:
                print("🔄 Running full event synchronization...")
                # Get both upcoming and updated events
                upcoming_events = self.call_upcoming_events_api()
                upcoming_dicts = [e.to_dict() for e in upcoming_events]

                updated_events = self.call_updated_events_api()
                updated_dicts = [e.to_dict() for e in updated_events]

                # Combine and deduplicate events
                all_events = upcoming_dicts + updated_dicts
                unique_events = {e['event_id']: e for e in all_events}.values()
                
                # For full sync, clear existing and add all
                if unique_events:
                    success = self.chroma_manager.add_events_batch(unique_events, clear_existing=True)
                    if success:
                        print(f"✅ Successfully synchronized {len(unique_events)} events (full sync)")
                    else:
                        print("❌ Failed to add events to ChromaDB")
                else:
                    print("ℹ️ No events found in API responses")
            else:
                print("🔄 Running incremental event synchronization...")
                # Only get updated events for incremental sync
                updated_events = self.call_updated_events_api()
                updated_dicts = [e.to_dict() for e in updated_events]

                # Add new/updated events without clearing existing
                if updated_dicts:
                    success = self.chroma_manager.add_events_batch(updated_dicts, clear_existing=False)
                    if success:
                        print(f"✅ Successfully synchronized {len(updated_dicts)} updated events")
                    else:
                        print("❌ Failed to add updated events to ChromaDB")
                else:
                    print("ℹ️ No updated events found")

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
        """Background sync loop - runs incremental updates"""
        while self.is_running:
            # Use incremental sync (only updated events) for periodic updates
            self.run_single_sync(full_sync=False)
            time.sleep(interval_minutes * 60)

class UserPreferenceSyncManager:
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager
        self.is_running = False
        self.sync_thread = None


    def fetch_user_preferences_page(self, cursor: int = None) -> tuple:
        print("ℹ️ User preferences are now managed via CSV files only.", flush=True)
        print("ℹ️ Please use import_user_preferences_from_csv_path() or prompt_and_import_csv_interactive() methods.", flush=True)
        return [], None

    def run_full_sync(self, clear_existing: bool = False):
        print("ℹ️ User preferences sync is now handled via CSV import only.", flush=True)
        print("ℹ️ Use import_user_preferences_from_csv_path() to import from CSV files.", flush=True)
        return

    def start_periodic_sync(self, interval_hours: int = 24, clear_existing_first_run: bool = False):
        """CSV-based user preferences don't require periodic sync."""
        print("ℹ️ Periodic sync not needed for CSV-based user preferences.", flush=True)
        print("ℹ️ Import new CSV files as needed using import methods.", flush=True)
        return

    def stop_periodic_sync(self):
        """Stop periodic synchronization"""
        self.is_running = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join()
        print("🛑 Stopped user preferences periodic sync")

    def import_user_preferences_from_dataframe(self, df: 'pd.DataFrame') -> int:
        """Import user preferences from a pandas DataFrame and upsert into ChromaDB."""
        try:
            print(f"🔧 DEBUG: Starting import_user_preferences_from_dataframe", flush=True)
            if df is None:
                print("❌ DEBUG: DataFrame is None", flush=True)
                return 0
            if df.empty:
                print("❌ DEBUG: DataFrame is empty", flush=True)
                return 0
            
            print(f"🔧 DEBUG: DataFrame shape: {df.shape}", flush=True)
            print(f"🔧 DEBUG: DataFrame columns: {list(df.columns)}", flush=True)
            
            # Clean column names - remove BOM and extra whitespace
            df.columns = [str(col).strip().replace('\ufeff', '').replace('﻿', '') for col in df.columns]
            print(f"🔧 DEBUG: Cleaned DataFrame columns: {list(df.columns)}", flush=True)
            
            print(f"🔧 DEBUG: First few rows:\n{df.head()}", flush=True)
            # Build case-insensitive column lookup
            lower_to_col = {str(c).strip().lower(): c for c in df.columns}

            def pick_col(*candidates: str):
                for cand in candidates:
                    key = cand.strip().lower()
                    if key in lower_to_col:
                        return lower_to_col[key]
                return None

            # Core columns based on actual CSV export structure
            id_col = pick_col('user id', 'user_id', 'userid', 'id', 'uid', '_id')  # Added: "user id" with space
            user_name_col = pick_col('user_name', 'username', 'display_name', 'name')
            first_name_col = pick_col('first name', 'first_name', 'firstname', 'first')  # Added: "first name" with space
            last_name_col = pick_col('last name', 'last_name', 'lastname', 'last', 'surname')  # Added: "last name" with space
            user_city_col = pick_col('user city', 'user_city', 'current_city', 'city', 'city_name')  # Added: "user city" with space
            # Activity row columns
            club_name_col = pick_col('club name', 'club_name', 'club', 'group', 'group_name', 'community', 'team')  # Added: "club name" with space
            club_id_col = pick_col('club id', 'club_id', 'clubid')  # Added: "club id" with space
            activity_col = pick_col('activity', 'category')
            activity_city_col = pick_col('club city', 'club_city', 'activity_city', 'city', 'city_name')  # Added: "club city" with space
            areas_col = pick_col('areas', 'area', 'locality', 'neighborhood')
            meetup_attended_col = pick_col('attended event count', 'attended_event_count', 'meetup_attended', 'attended', 'count')  # Added: "attended event count" with spaces
            # Optional direct summary and structured activities JSON
            activities_summary_col = pick_col('activities_summary')
            activities_json_col = pick_col('user_activities', 'activities')
            
            print(f"🔧 DEBUG: Column mappings:", flush=True)
            print(f"  id_col: {id_col} ('{id_col}' found in columns)" if id_col else f"  id_col: None (not found)", flush=True)
            print(f"  first_name_col: {first_name_col}", flush=True)
            print(f"  last_name_col: {last_name_col}", flush=True)
            print(f"  user_city_col: {user_city_col}", flush=True)
            print(f"  club_name_col: {club_name_col}", flush=True)
            print(f"  club_id_col: {club_id_col}", flush=True)
            print(f"  activity_col: {activity_col}", flush=True)
            print(f"  activity_city_col: {activity_city_col}", flush=True)
            print(f"  areas_col: {areas_col}", flush=True)
            print(f"  meetup_attended_col: {meetup_attended_col}", flush=True)
            
            print(f"🔧 DEBUG: Available column names in lowercase: {list(lower_to_col.keys())}", flush=True)

            # Group by user_id, aggregating activities across rows
            grouped: Dict[str, dict] = {}
            processed_rows = 0
            skipped_rows = 0
            
            print(f"🔧 DEBUG: Starting to process {len(df)} rows", flush=True)
            
            for idx, row in df.iterrows():
                processed_rows += 1
                # user_id
                if id_col is None:
                    print(f"❌ DEBUG: Row {idx}: No user_id column found, skipping", flush=True)
                    skipped_rows += 1
                    continue
                v = row.get(id_col)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    print(f"❌ DEBUG: Row {idx}: Empty user_id, skipping", flush=True)
                    skipped_rows += 1
                    continue
                user_id = str(v)
                
                if processed_rows <= 3:  # Only log first few rows to avoid spam
                    print(f"🔧 DEBUG: Row {idx}: Processing user_id={user_id}", flush=True)

                # Initialize base record
                rec = grouped.get(user_id)
                if rec is None:
                    # user_name
                    user_name = None
                    if user_name_col is not None:
                        nv = row.get(user_name_col)
                        user_name = None if pd.isna(nv) else str(nv)
                    if not user_name:
                        first_part = None
                        last_part = None
                        if first_name_col is not None:
                            fv = row.get(first_name_col)
                            first_part = None if pd.isna(fv) else str(fv)
                        if last_name_col is not None:
                            lv = row.get(last_name_col)
                            last_part = None if pd.isna(lv) else str(lv)
                        name_parts = [p for p in [first_part, last_part] if p]
                        if name_parts:
                            user_name = " ".join(name_parts)

                    # user-level city
                    current_city = None
                    if user_city_col is not None:
                        cv = row.get(user_city_col)
                        current_city = None if pd.isna(cv) else str(cv)

                    # direct activities_summary if present
                    activities_summary_direct = None
                    if activities_summary_col is not None:
                        sv = row.get(activities_summary_col)
                        activities_summary_direct = None if pd.isna(sv) else str(sv)

                    # structured activities JSON if present (rare)
                    structured_acts: List[dict] = []
                    if activities_json_col is not None:
                        jv = row.get(activities_json_col)
                        if isinstance(jv, str) and jv.strip():
                            try:
                                parsed = json.loads(jv)
                                if isinstance(parsed, list):
                                    structured_acts = parsed
                            except Exception:
                                structured_acts = []
                        elif isinstance(jv, list):
                            structured_acts = jv

                    rec = {
                        'user_id': user_id,
                        'user_name': user_name,
                        'current_city': current_city,
                        'activities_summary': activities_summary_direct or "",
                        'user_activities': structured_acts or [],
                        '_segments': []  # internal aggregation for summary
                    }
                    grouped[user_id] = rec

                # Per-row activity aggregation (long-form CSV)
                club_name = None
                if club_name_col is not None:
                    cn = row.get(club_name_col)
                    if cn is not None and not pd.isna(cn):
                        club_name = str(cn)
                club_id = None
                if club_id_col is not None:
                    cid = row.get(club_id_col)
                    if cid is not None and not pd.isna(cid):
                        club_id = str(cid)
                activity = None
                if activity_col is not None:
                    ac = row.get(activity_col)
                    if ac is not None and not pd.isna(ac):
                        activity = str(ac)
                act_city = None
                if activity_city_col is not None:
                    c2 = row.get(activity_city_col)
                    if c2 is not None and not pd.isna(c2):
                        act_city = str(c2)
                # areas parse - handle PostgreSQL array format like [ "GCR Extn." ]
                areas_list: List[str] = []
                if areas_col is not None:
                    av = row.get(areas_col)
                    if isinstance(av, str) and av.strip():
                        # Handle PostgreSQL array format: [ "GCR Extn." ]
                        av_clean = av.strip()
                        if av_clean == '[]':
                            # Empty array
                            areas_list = []
                        elif av_clean.startswith('[') and av_clean.endswith(']'):
                            try:
                                # Try JSON parsing first
                                parsed_areas = json.loads(av_clean)
                                if isinstance(parsed_areas, list):
                                    areas_list = [str(a).strip() for a in parsed_areas if a is not None]
                                else:
                                    areas_list = [str(av_clean)]
                            except Exception:
                                # If JSON parsing fails, try manual PostgreSQL array parsing
                                try:
                                    # Remove brackets and split by comma
                                    inner = av_clean[1:-1].strip()
                                    if inner:
                                        # Split and clean each item
                                        items = []
                                        for item in inner.split(','):
                                            item = item.strip()
                                            # Remove quotes if present
                                            if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                                                item = item[1:-1]
                                            if item:
                                                items.append(item)
                                        areas_list = items
                                    else:
                                        areas_list = []
                                except Exception:
                                    areas_list = [str(av_clean)]
                        else:
                            # Not array format, treat as single item
                            areas_list = [str(av_clean)]
                    elif isinstance(av, list):
                        areas_list = [str(a) for a in av if a is not None]
                    
                    if processed_rows <= 3:  # Debug log for first few rows
                        print(f"🔧 DEBUG: Row {idx}: areas raw='{av}' -> parsed={areas_list}", flush=True)
                # attended
                attended_val = None
                if meetup_attended_col is not None:
                    mv = row.get(meetup_attended_col)
                    try:
                        attended_val = int(mv) if mv is not None and not pd.isna(mv) else None
                    except Exception:
                        attended_val = None

                # Handle different cases for club data
                if club_id and club_id.lower() not in ['null', 'none', ''] and club_name and club_name.lower() not in ['null', 'none', '']:
                    # Valid club data - add activity entry
                    activity_entry = {
                        'club_id': club_id or "",
                        'club_name': club_name or "",
                        'activity': activity or "",
                        'city': act_city or (rec.get('current_city') or ""),
                        'area': areas_list,
                        'meetup_attended': attended_val if attended_val is not None else ""
                    }
                    rec['user_activities'].append(activity_entry)
                    # Build segment for summary
                    areas_text = ", ".join(areas_list) if areas_list else ""
                    seg = f"{activity_entry['club_name']}|{activity_entry['activity']}|{activity_entry['city']}|{areas_text}|{activity_entry['meetup_attended']}"
                    rec['_segments'].append(seg)
                    
                    if processed_rows <= 3:
                        print(f"🔧 DEBUG: Row {idx}: Added activity for user {user_id}: {activity_entry['club_name']}", flush=True)
                elif processed_rows <= 3:
                    # User has no valid club data - this is OK, just log it
                    print(f"🔧 DEBUG: Row {idx}: User {user_id} has no valid club data (club_id='{club_id}', club_name='{club_name}')", flush=True)

            # Finalize records from grouped dict
            records: List[dict] = []
            print(f"🔧 DEBUG: Finalizing records from {len(grouped)} grouped users", flush=True)
            
            for user_id, rec in grouped.items():
                if not rec.get('activities_summary') and rec.get('_segments'):
                    rec['activities_summary'] = " ; ".join([s for s in rec['_segments'] if s])
                rec.pop('_segments', None)
                records.append(rec)
                
                if len(records) <= 3:  # Only log first few records
                    print(f"🔧 DEBUG: User {user_id} record: {rec}", flush=True)
            
            print(f"🔧 DEBUG: Processing summary:", flush=True)
            print(f"  Total rows processed: {processed_rows}", flush=True)
            print(f"  Rows skipped: {skipped_rows}", flush=True)
            print(f"  Final records created: {len(records)}", flush=True)

            if not records:
                print("❌ DEBUG: No valid records found to import from DataFrame", flush=True)
                return 0
            
            print(f"🔧 DEBUG: Calling add_user_preferences_batch with {len(records)} records", flush=True)
            ok = self.chroma_manager.add_user_preferences_batch(records, clear_existing=False)
            print(f"🔧 DEBUG: add_user_preferences_batch returned: {ok}", flush=True)
            
            return len(records) if ok else 0
        except Exception as e:
            print(f"❌ Error importing user preferences from DataFrame: {e}", flush=True)
            return 0

    def import_user_preferences_from_csv_path(self, csv_path: str) -> int:
        """Import user preferences from a CSV file path and upsert into ChromaDB."""
        try:
            if not csv_path or not os.path.exists(csv_path):
                print(f"❌ CSV path not found: {csv_path}", flush=True)
                return 0
            print(f"🔧 DEBUG: Reading CSV file: {csv_path}", flush=True)
            # Use encoding='utf-8-sig' to handle BOM character properly
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            return self.import_user_preferences_from_dataframe(df)
        except Exception as e:
            print(f"❌ Error reading CSV at {csv_path}: {e}", flush=True)
            return 0

    def prompt_and_import_csv_interactive(self) -> int:
        """Prompt user to upload or provide path to CSV and import user preferences."""
        try:
            # Prefer Colab's upload UI if available
            try:
                from google.colab import files as colab_files  # type: ignore
                print("📤 Please upload a CSV file with user preferences...", flush=True)
                uploaded = colab_files.upload()
                if uploaded:
                    name = next(iter(uploaded.keys()))
                    data = uploaded[name]
                    df = pd.read_csv(io.BytesIO(data))
                    count = self.import_user_preferences_from_dataframe(df)
                    print(f"✅ Imported {count} user preferences from uploaded CSV", flush=True)
                    return count
            except Exception:
                # Not in Colab or upload failed; fall back to path prompt
                pass

            try:
                path = input("Enter CSV path to import user preferences (or press Enter to skip): ").strip()
            except Exception:
                path = ""
            if not path:
                print("ℹ️ CSV import skipped", flush=True)
                return 0
            count = self.import_user_preferences_from_csv_path(path)
            if count > 0:
                print(f"✅ Imported {count} user preferences from {path}", flush=True)
            return count
        except Exception as e:
            print(f"❌ Error during interactive CSV import: {e}", flush=True)
            return 0

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

            # Try multiple URL field names for registration
            registration_url = (
                event.get('registration_url') or 
                event.get('event_url') or 
                event.get('signup_url') or 
                event.get('booking_url') or 
                event.get('link') or 
                event.get('url')
            )
            if registration_url:
                response += f"🔗 **Register Here**: {registration_url}\n"
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
                # Check for explicit preference phrases
                preference_pattern = r"\b(prefer|like|love|interested|into|my (hobbies|interests)|i want|i would like|i enjoy)\b"
                if re.search(preference_pattern, text, flags=re.IGNORECASE):
                    return True
                
                # Check for specific activities mentioned
                activities = ['football', 'cricket', 'badminton', 'tennis', 'swimming', 'gym', 'yoga', 
                            'dance', 'music', 'art', 'photography', 'hiking', 'trekking', 'cycling',
                            'tech', 'coding', 'startup', 'business', 'networking', 'food', 'cooking',
                            'basketball', 'volleyball', 'chess', 'gaming', 'reading', 'writing']
                
                for activity in activities:
                    if re.search(rf"\b{activity}\b", text, flags=re.IGNORECASE):
                        return True
                
                # Check for location preferences
                location_pattern = r"\b(in|at|near|around)\s+[A-Za-z]+\b"
                if re.search(location_pattern, text, flags=re.IGNORECASE):
                    return True
                
                # Check for budget mentions
                budget_pattern = r"(₹\s*\d+|budget|price|cost|under|below|above)"
                if re.search(budget_pattern, text, flags=re.IGNORECASE):
                    return True
                
                # Check for timing preferences
                timing_pattern = r"\b(weekend|weekday|morning|afternoon|evening|night|today|tomorrow|this week)\b"
                if re.search(timing_pattern, text, flags=re.IGNORECASE):
                    return True
                
                return False
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

        # Check if this is the first message without preferences
        is_initial_request = len(self.conversation_history) == 0
        
        # Extract specific preferences from current message
        def _extract_message_preferences(text: str) -> dict:
            prefs = {}
            
            # Extract activities
            activities = ['football', 'cricket', 'badminton', 'tennis', 'swimming', 'gym', 'yoga', 
                        'dance', 'music', 'art', 'photography', 'hiking', 'trekking', 'cycling',
                        'tech', 'coding', 'startup', 'business', 'networking', 'food', 'cooking',
                        'basketball', 'volleyball', 'chess', 'gaming', 'reading', 'writing']
            
            found_activities = []
            for activity in activities:
                if re.search(rf"\b{activity}\b", text, flags=re.IGNORECASE):
                    found_activities.append(activity)
            if found_activities:
                prefs['activities'] = found_activities
            
            # Extract location
            location_match = re.search(r'\b(?:in|at|near|around)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b', text, re.IGNORECASE)
            if location_match:
                prefs['location'] = location_match.group(1)
            
            # Extract timing
            timing_patterns = {
                'weekend': r'\bweekend\b',
                'weekday': r'\bweekday\b',
                'morning': r'\bmorning\b',
                'afternoon': r'\bafternoon\b',
                'evening': r'\bevening\b',
                'night': r'\bnight\b',
                'today': r'\btoday\b',
                'tomorrow': r'\btomorrow\b',
                'this week': r'\bthis week\b'
            }
            
            for timing, pattern in timing_patterns.items():
                if re.search(pattern, text, flags=re.IGNORECASE):
                    prefs['timing'] = timing
                    break
            
            # Extract budget
            budget_match = re.search(r'₹?\s*(\d+)', text)
            if budget_match:
                prefs['budget'] = budget_match.group(1)
            
            return prefs
        
        message_prefs = _extract_message_preferences(user_message)
        
        # Decide whether to search events now or ask for preferences first
        relevant_events = []
        should_ask_preferences = False
        
        # Search for events based on preferences
        if not prefs_missing:  # User preferences FOUND - use them!
            # Build search query from stored preferences
            search_query = user_message
            if user_pref_context and "activities_summary" in user_pref_context:
                # Extract activities from the preference context to enhance search
                import re
                activities = re.findall(r'activities_summary: ([^\n]+)', user_pref_context)
                if activities and activities[0] != 'N/A':
                    search_query = f"{user_message} {activities[0]}"
            relevant_events = self.search_events_vector(search_query, n_results=10)
        elif message_prefs:  # User provided preferences in current message
            # Use preferences from current message
            relevant_events = self.search_events_vector(user_message, n_results=10)
        elif _message_has_preference_clues(user_message):  # User provided some preferences in message
            relevant_events = self.search_events_vector(user_message, n_results=10)
        elif is_initial_request and prefs_missing:  # First message, no preferences at all
            should_ask_preferences = True

        # Convert relevant events to string format for context
        events_context = "Relevant Events Data (from vector search):\n"
        
        if should_ask_preferences:
            events_context += "STATUS: No user preferences found - MUST ASK FOR PREFERENCES FIRST\n"
            events_context += "ACTION REQUIRED: Ask the user about their preferences before showing any events.\n"
        elif not prefs_missing and relevant_events:
            events_context += "STATUS: USER PREFERENCES FOUND - SHOW PERSONALIZED RECOMMENDATIONS\n"
            events_context += "ACTION REQUIRED: Use the found preferences to recommend relevant events with enthusiasm.\n\n"
            for event in relevant_events:
                events_context += f"🎯 Event: {event.get('name', 'N/A')}\n"
                events_context += f"  Club: {event.get('club_name', 'N/A')}\n"
                events_context += f"  Activity: {event.get('activity', 'N/A')}\n"
                events_context += f"  When: {event.get('start_time', 'N/A')} to {event.get('end_time', 'N/A')}\n"
                events_context += f"  Where: {event.get('location_name', 'N/A')} ({event.get('area_name', 'N/A')}, {event.get('city_name', 'N/A')})\n"
                events_context += f"  Price: ₹{event.get('ticket_price', 'N/A')} | Spots: {event.get('available_spots', 'N/A')}\n"
                events_context += f"  Payment: {event.get('payment_terms', 'N/A')}\n"
                # Try multiple URL field names
                event_url = (
                    event.get('registration_url') or 
                    event.get('event_url') or 
                    event.get('signup_url') or 
                    event.get('booking_url') or 
                    event.get('link') or 
                    event.get('url') or 'N/A'
                )
                # Debug: Show what URL fields are available
                # url_debug = {k: event.get(k) for k in ['registration_url', 'event_url', 'signup_url', 'booking_url', 'link', 'url'] if event.get(k)}
                # if url_debug:
                #     print(f"🔧 Debug URLs for {event.get('name', 'Unknown')}: {url_debug}")
                
                events_context += f"  REGISTRATION_URL: {event_url}\n\n"
        elif relevant_events:
            if message_prefs:
                events_context += "STATUS: USER PROVIDED PREFERENCES IN MESSAGE - SHOW RELEVANT EVENTS\n"
                events_context += f"Detected preferences from message: {message_prefs}\n\n"
            else:
                events_context += "STATUS: USER PROVIDED PREFERENCES IN MESSAGE - SHOW RELEVANT EVENTS\n\n"
            for event in relevant_events:
                events_context += f"🎯 Event: {event.get('name', 'N/A')}\n"
                events_context += f"  Club: {event.get('club_name', 'N/A')}\n"
                events_context += f"  Activity: {event.get('activity', 'N/A')}\n"
                events_context += f"  When: {event.get('start_time', 'N/A')} to {event.get('end_time', 'N/A')}\n"
                events_context += f"  Where: {event.get('location_name', 'N/A')} ({event.get('area_name', 'N/A')}, {event.get('city_name', 'N/A')})\n"
                events_context += f"  Price: ₹{event.get('ticket_price', 'N/A')} | Spots: {event.get('available_spots', 'N/A')}\n"
                events_context += f"  Payment: {event.get('payment_terms', 'N/A')}\n"
                # Try multiple URL field names
                event_url = (
                    event.get('registration_url') or 
                    event.get('event_url') or 
                    event.get('signup_url') or 
                    event.get('booking_url') or 
                    event.get('link') or 
                    event.get('url') or 'N/A'
                )
                # Debug: Show what URL fields are available
                # url_debug = {k: event.get(k) for k in ['registration_url', 'event_url', 'signup_url', 'booking_url', 'link', 'url'] if event.get(k)}
                # if url_debug:
                #     print(f"🔧 Debug URLs for {event.get('name', 'Unknown')}: {url_debug}")
                
                events_context += f"  REGISTRATION_URL: {event_url}\n\n"
        else:
            if prefs_missing and not _message_has_preference_clues(user_message):
                events_context += "STATUS: No preferences found and user hasn't provided any.\n"
                events_context += "ACTION REQUIRED: Ask the user about their hobbies, interests, preferred activities, city/area, budget, and timing.\n"
            else:
                events_context += "No specific events found for this query. Consider asking user to provide more specific preferences.\n"

        # Add conversation history
        history_context = "\nConversation History:\n"
        for msg in self.conversation_history[-6:]:
            history_context += f"{msg['role']}: {msg['content']}\n"

        system_prompt = f"""You are a warm, enthusiastic, and friendly meetup recommendation assistant named Miffy. You love helping people discover exciting events and make new connections. You have access to a vector database of events and provide personalized recommendations with genuine enthusiasm.

{events_context}
{history_context}
{user_pref_context}

Instructions:

PERSONALITY & TONE:
- Be warm, enthusiastic, and conversational (not robotic)
- Use varied greetings and expressions each time
- Show genuine excitement about events you recommend
- Add personality with phrases like "Oh, this looks perfect for you!" or "You're going to love this one!"
- Keep it friendly but not overly casual

🎯 CRITICAL PREFERENCE HANDLING WORKFLOW:

**LOCATION PRIORITY RULE:**
🚨 **IMPORTANT**: If user provides current_city (e.g., Gurugram), ONLY show events in that city!
- NEVER recommend events from other cities unless user explicitly asks
- If user is in Gurugram, don't show Mumbai/Delhi events
- Only exception: User explicitly says "show me events in [other city]" or "anywhere"

**STEP 1: ALWAYS CHECK USER PREFERENCES FIRST**
When a user requests event recommendations with their user_id:

A. **USER_ID PROVIDED + SAVED PREFERENCES FOUND:**
   - ✅ IMMEDIATELY use their saved preferences to find the BEST matching events
   - Greet warmly: "Hey [Name]! Based on your saved preferences for [activity] in [location], here are some perfect events for you!"
   - Show 3-7 highly relevant events that match their stored interests
   - NO NEED to ask for preferences - they're already saved and loaded

B. **USER_ID PROVIDED + NO SAVED PREFERENCES:**
   - ⚠️ MUST ask for preferences FIRST before showing any events
   - Friendly approach: "Hi! I'd love to find the perfect events for you! Since this is our first time, let me learn about your interests:
     • What activities get you excited? (sports, arts, tech, etc.)
     • Which area/city works best for you?
     • Any budget range I should keep in mind?
     • Do you prefer weekends or weekday evenings?"
   - WAIT for their response - DO NOT show random events
   - Once they provide preferences, SAVE them and then recommend matching events

C. **PREFERENCES PROVIDED IN CURRENT MESSAGE (with or without user_id):**
   - Enthusiastically acknowledge: "Great! A [activity] lover in [location]! Let me find amazing [activity] events for you!"
   - Use those preferences IMMEDIATELY to search for relevant events
   - Smart handling - DON'T ask for info they already gave:
     ❌ If they said "badminton" - don't ask about activities
     ❌ If they said "Mumbai" - don't ask about location  
     ❌ If they mentioned budget - don't ask about price
   - Only ask for missing details that would improve recommendations

D. **NO USER_ID + NO PREFERENCES:**
   - Ask for preferences first in a warm, conversational way

📋 MANDATORY RECOMMENDATION WORKFLOW:

**PREFERENCE-FIRST APPROACH:**
1. **ALWAYS check for user preferences FIRST** (saved in database or mentioned in current message)
2. **If user_id provided:**
   - Look up their saved preferences immediately
   - If found → Use them to get the BEST matching events
   - If not found → Ask for preferences BEFORE showing any events
3. **If preferences found or provided:** Search with enthusiasm: "Let me find some amazing [activity] events for you in [location]!"
4. **Return 3-7 highly relevant events** that match their specific interests, not generic events
5. **NEVER show random/generic events** - always preference-matched events or ask for preferences first

**EVENT PRESENTATION REQUIREMENTS:**
6. For each recommended event, present with excitement and ALWAYS include:
   - Name and organizing club ("This looks perfect!" / "You'll love this one!")
   - Activity type with enthusiasm
   - Date and time (local) in IST
   - **LOCATION DETAILS (MANDATORY):** Include full venue name, area, and city - users need to know exactly where to go!
   - Price and available spots
   - Why this matches their interests (1-2 lines)
   - **CRITICAL: Registration URL (exact from database) - MUST BE INCLUDED FOR EVERY EVENT**
6. If no exact matches: "I found some similar amazing options you might enjoy!"
7. Use varied expressions: "Check this out!" / "How about this?" / "This could be great!"
8. Add helpful tips: "Book soon, spots filling up!" / "Perfect for beginners!" 
9. Close with engagement: "Which one catches your eye?" / "Want to see more options?"
10. Always retrieve event data from vector database only

**MANDATORY URL REQUIREMENT:**
- EVERY single event recommendation MUST include the registration URL
- Format: "🔗 **Register:** " followed by the plain URL on its own line
- Example: "🔗 **Register:** https://example.com/event123"
- Use the exact URL provided in the event data (listed as "REGISTRATION_URL: ..." in each event)
- DO NOT use markdown link format [text](url) - use plain URLs only
- If URL shows "N/A", mention "Registration details available on request"
- NEVER skip the registration link - users need this to book events!

⚠️ CRITICAL REMINDERS: 
- **PREFERENCE-FIRST:** Always check for saved user preferences when user_id is provided
- **NO GENERIC EVENTS:** Never show random events - only preference-matched or ask for preferences first
- Each event MUST include complete location details (venue, area, city) - users need to know where to go!
- Each event MUST end with its registration URL as a plain, clickable link. No exceptions!

🎯 **SUMMARY:** With user_id → Check preferences → If found: use them, If not found: ask first → Then recommend matching events

Current user message: {user_message}"""
        return system_prompt

    def extract_and_save_user_preferences(self, user_message: str, user_id: str = None):
        """Extract preferences from user message and optionally save them"""
        preferences = {}
        
        # Extract city/location
        city_match = re.search(r'\b(?:in|at|near|around)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b', user_message, re.IGNORECASE)
        if city_match:
            preferences['city'] = city_match.group(1)
        
        # Extract activities/interests
        activity_keywords = ['football', 'cricket', 'badminton', 'tennis', 'swimming', 'gym', 'yoga', 
                           'dance', 'music', 'art', 'photography', 'hiking', 'trekking', 'cycling',
                           'tech', 'coding', 'startup', 'business', 'networking', 'food', 'cooking']
        found_activities = []
        for activity in activity_keywords:
            if activity.lower() in user_message.lower():
                found_activities.append(activity)
        if found_activities:
            preferences['activities'] = found_activities
        
        # Extract budget
        budget_match = re.search(r'₹?\s*(\d+)(?:\s*-\s*₹?\s*(\d+))?', user_message)
        if budget_match:
            preferences['budget_min'] = int(budget_match.group(1))
            if budget_match.group(2):
                preferences['budget_max'] = int(budget_match.group(2))
        
        # Extract time preferences
        time_keywords = {
            'morning': 'morning',
            'afternoon': 'afternoon', 
            'evening': 'evening',
            'night': 'night',
            'weekend': 'weekend',
            'weekday': 'weekday'
        }
        for keyword, value in time_keywords.items():
            if keyword in user_message.lower():
                preferences['preferred_time'] = value
                break
        
        return preferences
    
    def generate_personalized_greeting(self, user_id: str = None, user_name: str = None):
        """Generate a personalized, friendly greeting that varies each time"""
        import random
        from datetime import datetime
        
        # Get current time of day
        hour = datetime.now().hour
        if hour < 12:
            time_greeting = "Good morning"
        elif hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
        
        # Try to get user preferences if user_id provided
        user_prefs = None
        user_activities = []
        user_city = None
        
        if user_id:
            try:
                user_prefs = self.chroma_manager.get_user_preferences_by_user_id(user_id)
                if user_prefs:
                    metadata = user_prefs.get('metadata', {})
                    user_name = user_name or metadata.get('user_name', '')
                    user_city = metadata.get('current_city', '')
                    activities_summary = metadata.get('activities_summary', '')
                    # Extract main activities from summary
                    if activities_summary:
                        for activity in ['football', 'cricket', 'tech', 'music', 'dance', 'yoga', 'hiking', 'food']:
                            if activity.lower() in activities_summary.lower():
                                user_activities.append(activity)
            except:
                pass
        
        # Base greetings with variety - Miffy style
        if user_name:
            name_greetings = [
                f"Hey {user_name}! It's Miffy here! 🌟",
                f"{time_greeting}, {user_name}! Miffy at your service! ✨",
                f"Welcome back, {user_name}! Miffy missed you! 🎉",
                f"Great to see you, {user_name}! It's your pal Miffy! 👋",
                f"Hello {user_name}! Miffy's excited to help! 🙌",
                f"Hi there, {user_name}! Miffy's ready for adventure! 😊"
            ]
            greeting = random.choice(name_greetings)
        else:
            generic_greetings = [
                f"{time_greeting}! I'm Miffy, your event companion! 🌟",
                "Hey there! Miffy here to help! ✨",
                "Welcome! I'm Miffy, let's find you something amazing! 🎉",
                "Hello friend! Miffy's ready to discover events with you! 👋",
                "Hi! I'm Miffy, and I'm excited to help you! 🙌",
                "Greetings! Miffy at your service! 😊"
            ]
            greeting = random.choice(generic_greetings)
        
        # Add personalized event teasers based on preferences
        if user_activities:
            activity = random.choice(user_activities)
            event_teasers = [
                f"I've spotted some amazing {activity} events that might interest you!",
                f"There are some cool {activity} meetups happening soon!",
                f"I found exciting {activity} activities waiting for you!",
                f"Some awesome {activity} events just popped up!",
                f"Fresh {activity} opportunities are available!",
                f"New {activity} adventures are calling your name!"
            ]
        elif user_city:
            event_teasers = [
                f"I've found some exciting events happening in {user_city}!",
                f"There are cool meetups waiting for you in {user_city}!",
                f"Some amazing activities are lined up in {user_city}!",
                f"Fresh events just dropped in {user_city}!",
                f"Your city {user_city} has some great events coming up!"
            ]
        else:
            event_teasers = [
                "I've discovered some exciting events you might love!",
                "There are amazing meetups waiting to be explored!",
                "Some cool activities are ready for you to discover!",
                "Fresh events and adventures are available!",
                "Interesting opportunities are waiting for you!",
                "New experiences are just a click away!"
            ]
        
        # Combine greeting with teaser
        full_greeting = f"{greeting} {random.choice(event_teasers)}"
        
        # Add call to action
        cta_options = [
            "\n\n🔍 What kind of adventure are you looking for today?",
            "\n\n🎯 What interests you today?",
            "\n\n🌈 What would you like to explore?",
            "\n\n✨ What sounds fun to you?",
            "\n\n🚀 Ready to discover something new?",
            "\n\n💫 What's on your mind today?"
        ]
        
        full_greeting += random.choice(cta_options)
        
        return full_greeting
    
    def get_bot_response(self, user_message):
        """Get response from the AI model with streaming"""
        try:
            # Check if this is the first message and handle user ID requests appropriately
            if len(self.conversation_history) == 0:
                # Try to extract user_id from message if provided
                user_id = None
                user_name = None
                is_user_id_request = False
                try:
                    import re
                    # Check if this is a user ID request
                    if re.search(r"(?:find events?|events?)\s+for\s+user[_ ]?id\s+(\d+)", user_message, re.IGNORECASE):
                        is_user_id_request = True
                        # Don't show greeting for user ID requests - the AI will handle it contextually
                    
                    id_match = re.search(r"(?:user[_ ]?id|uid|id)[:\s]*(\d+)", user_message, re.IGNORECASE)
                    if id_match:
                        user_id = id_match.group(1)
                    
                    # Try to extract name
                    name_match = re.search(r"(?:i'm|i am|name is|this is)\s+([A-Z][a-z]+)", user_message, re.IGNORECASE)
                    if name_match:
                        user_name = name_match.group(1)
                except:
                    pass
                
                # Only show greeting if it's not a user ID request
                if not is_user_id_request:
                    greeting = self.generate_personalized_greeting(user_id, user_name)
                    print(f"\nBot: {greeting}\n")
            
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
            print("Miffy: ", end="", flush=True)
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
        existing_events = collection_info.get('total_events', 0)
        
        if existing_events == 0:
            print("❌ No event data available. Running full initial sync...")
            self.event_sync_manager.run_single_sync(full_sync=True)
            time.sleep(2)  # Wait a moment for sync to complete

            # Check again after sync
            collection_info = self.chroma_manager.get_collection_stats()
            if collection_info.get('total_events', 0) == 0:
                print("❌ Still no events found after sync. Please check API connection.")
                return
        else:
            print(f"✅ Found {existing_events} existing events. Ready to start!")

        print("🤖 Miffy is ready to help you discover amazing events! Type 'quit' to exit.\n")
        print("=" * 50)
        
        # Generate initial greeting
        initial_greeting = self.generate_personalized_greeting()
        print(f"\n🎉 Miffy: {initial_greeting}")
        print("\n💡 **Quick Start**: You can say things like:")
        print("   • 'I'm John and I love football'")
        print("   • 'Show me tech events in Mumbai under ₹500'")
        print("   • 'I enjoy hiking and outdoor activities'")
        print("   • 'Looking for weekend activities'")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    farewell_messages = [
                        "Awesome chatting with you! Miffy hopes you find amazing events! 🎊",
                        "See you soon! Miffy can't wait to help you discover more meetups! 👋",
                        "Take care! Miffy hopes you have a blast at your next event! 🌟",
                        "Goodbye friend! Miffy wishes you amazing adventures ahead! ✨",
                        "Until next time! Miffy says: May your events be fun and friends be many! 🎉"
                    ]
                    import random
                    print(f"Miffy: {random.choice(farewell_messages)}")
                    break
                if not user_input:
                    continue
                # Get and display bot response
                self.get_bot_response(user_input)
            except KeyboardInterrupt:
                print("\n\nMiffy: Goodbye! Hope you find some amazing meetups! 👋")
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

    def sync_events_once(self, full_sync: bool = False):
        """Run a single event synchronization cycle
        
        Args:
            full_sync: If True, performs full sync (default: False for incremental)
        """
        self.event_sync_manager.run_single_sync(full_sync=full_sync)

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

    def _calculate_match_score(self, event: dict, preferences: dict) -> float:
        """Calculate how well an event matches user preferences (0.0 to 1.0)"""
        score = 0.0
        factors = 0
        
        # Activity match (40% weight)
        user_activities = preferences.get('activities', [])
        event_activity = event.get('activity', '').lower()
        if user_activities:
            for activity in user_activities:
                if activity.lower() in event_activity or event_activity in activity.lower():
                    score += 0.4
                    break
            factors += 1
        
        # Location match (30% weight)
        user_location = preferences.get('location', '').lower()
        user_areas = [area.lower() for area in preferences.get('areas', [])]
        event_area = event.get('area_name', '').lower()
        event_city = event.get('city_name', '').lower()
        
        if user_location:
            if user_location in event_city or event_city in user_location:
                score += 0.3
            elif user_areas and event_area in user_areas:
                score += 0.25
            factors += 1
        
        # Budget match (20% weight)
        user_budget = preferences.get('budget_max')
        event_price = event.get('ticket_price', 0)
        if user_budget is not None:
            if event_price <= user_budget:
                score += 0.2
            elif event_price <= user_budget * 1.2:  # 20% tolerance
                score += 0.1
            factors += 1
        
        # Availability bonus (10% weight)
        available_spots = event.get('available_spots', 0)
        if available_spots > 0:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

    def _generate_recommendation_reason(self, event: dict, preferences: dict, score: float) -> str:
        """Generate a human-readable reason why this event was recommended"""
        reasons = []
        
        # Activity match
        user_activities = preferences.get('activities', [])
        event_activity = event.get('activity', '')
        if user_activities and any(activity.lower() in event_activity.lower() for activity in user_activities):
            reasons.append(f"matches your interest in {event_activity}")
        
        # Location match  
        event_area = event.get('area_name', '')
        user_areas = preferences.get('areas', [])
        if user_areas and event_area.lower() in [area.lower() for area in user_areas]:
            reasons.append(f"in your preferred area ({event_area})")
        
        # Budget match
        user_budget = preferences.get('budget_max')
        event_price = event.get('ticket_price', 0)
        if user_budget and event_price <= user_budget:
            reasons.append(f"within your budget (₹{event_price})")
        
        # High match score
        if score > 0.8:
            reasons.append("excellent overall match")
        elif score > 0.6:
            reasons.append("good match for your preferences")
            
        return f"Perfect because it {', '.join(reasons)}" if reasons else "Recommended based on your preferences"

    def get_recommendations_json(self, request_data: dict) -> dict:
        """
        Main JSON API for getting personalized event recommendations
        Smart city filtering - works with ANY city (Delhi, Noida, Gurugram, Mumbai, etc.)
        
        Args:
            request_data: {
                "query": "user's natural language query",
                "user_id": "optional user id",
                "user_current_city": "user's current location (any city)",
                "limit": 5,
                "preferences": {...}
            }
        
        Returns:
            Structured JSON with recommendations filtered by current city
        """
        try:
            user_id = request_data.get('user_id')
            user_current_city = request_data.get('user_current_city', '').lower().strip()
            query = request_data.get('query', '')
            limit = request_data.get('limit', 5)
            override_preferences = request_data.get('preferences', {})
            
            # Get user preferences
            user_preferences = {}
            if user_id:
                saved_prefs = self.chroma_manager.get_user_preferences_by_user_id(user_id)
                if saved_prefs:
                    user_preferences = saved_prefs
            
            # Override with provided preferences
            if override_preferences:
                user_preferences.update(override_preferences)
            
            # IMPORTANT: Set current city as primary location filter
            if user_current_city:
                user_preferences['current_city'] = user_current_city
                # Unless user explicitly asks for another city, filter by current city
                if not any(city_keyword in query.lower() for city_keyword in ['mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata']):
                    user_preferences['location'] = user_current_city
            
            # Build search query
            search_query = query
            if not search_query and user_preferences:
                activities = user_preferences.get('activities', [])
                if activities:
                    search_query = ' '.join(activities)
                if user_current_city:
                    search_query += f" {user_current_city}"
            
            # Get events
            relevant_events = []
            if search_query:
                # Get more results initially for better filtering
                relevant_events = self.chroma_manager.search_events(search_query, n_results=limit * 3)
            
            # Filter by current city if provided - WORKS WITH ANY CITY
            if user_current_city:
                # Strict city filtering - only show events in current city
                city_filtered = []
                other_city_events = []
                
                # City name variations mapping (handles common variations)
                city_variations = {
                    'gurugram': ['gurugram', 'gurgaon'],
                    'gurgaon': ['gurugram', 'gurgaon'],
                    'delhi': ['delhi', 'new delhi'],
                    'new delhi': ['delhi', 'new delhi'],
                    'noida': ['noida'],
                    'mumbai': ['mumbai', 'bombay'],
                    'bombay': ['mumbai', 'bombay'],
                    'bangalore': ['bangalore', 'bengaluru'],
                    'bengaluru': ['bangalore', 'bengaluru'],
                    'chennai': ['chennai', 'madras'],
                    'kolkata': ['kolkata', 'calcutta'],
                    'pune': ['pune'],
                    'hyderabad': ['hyderabad'],
                    'ahmedabad': ['ahmedabad'],
                    'faridabad': ['faridabad'],
                    'ghaziabad': ['ghaziabad']
                }
                
                # Get variations for current city
                current_city_variations = city_variations.get(user_current_city, [user_current_city])
                
                for event in relevant_events:
                    event_city = event.get('city_name', '').lower().strip()
                    
                    # Check if event city matches any variation of current city
                    city_match = False
                    for variation in current_city_variations:
                        if variation in event_city or event_city in variation:
                            city_match = True
                            break
                    
                    # Also check exact match
                    if not city_match and (user_current_city in event_city or event_city in user_current_city):
                        city_match = True
                    
                    if city_match:
                        city_filtered.append(event)
                    else:
                        other_city_events.append(event)
                
                # Only use other city events if user explicitly asks or no local events found
                if city_filtered:
                    relevant_events = city_filtered
                elif 'anywhere' in query.lower() or 'any city' in query.lower():
                    relevant_events = other_city_events
                else:
                    # No events in current city
                    relevant_events = []
            
            # Score and format events
            scored_events = []
            for event in relevant_events:
                score = self._calculate_match_score_enhanced(event, user_preferences, user_current_city)
                
                # Get all possible registration URLs
                registration_url = (
                    event.get('event_url') or
                    event.get('registration_url') or
                    event.get('signup_url') or
                    event.get('booking_url') or
                    event.get('link') or
                    event.get('url') or
                    "Contact organizer for registration"
                )
                
                if score > 0.2:  # Lower threshold for better coverage
                    event_data = {
                        "event_id": event.get('event_id', ''),
                        "name": event.get('name', event.get('event_name', '')),
                        "club_name": event.get('club_name', ''),
                        "activity": event.get('activity', ''),
                        "start_time": event.get('start_time', ''),
                        "end_time": event.get('end_time', ''),
                        "location": {
                            "venue": event.get('location_name', ''),
                            "area": event.get('area_name', ''),
                            "city": event.get('city_name', ''),
                            "full_address": f"{event.get('location_name', '')}, {event.get('area_name', '')}, {event.get('city_name', '')}"
                        },
                        "price": event.get('ticket_price', 0),
                        "available_spots": event.get('available_spots', 0),
                        "registration_url": registration_url,
                        "_score": score  # Temporary for sorting
                    }
                    scored_events.append(event_data)
            
            # Sort by score (highest first)
            scored_events.sort(key=lambda x: x['_score'], reverse=True)
            
            # Remove internal score field from all events
            for event_data in scored_events:
                event_data.pop('_score', None)
            top_recommendations = scored_events[:limit]
            
            # Generate bot response message
            bot_response = self._generate_natural_response(top_recommendations, query, user_current_city)
            
            return {
                "success": True,
                "recommendations": top_recommendations,
                "total_found": len(scored_events),
                "user_preferences_used": user_preferences,
                "message": f"Found {len(top_recommendations)} events in {user_current_city}" if user_current_city else f"Found {len(top_recommendations)} recommended events",
                "bot_response": bot_response,
                "query": query,
                "user_current_city": user_current_city
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recommendations": [],
                "message": f"Error: {str(e)}"
            }

    def _calculate_match_score_enhanced(self, event: dict, preferences: dict, current_city: str = "") -> float:
        """Enhanced scoring with dynamic city preference - works with ANY city"""
        score = 0.0
        
        # City match (highest priority - 35% weight)
        if current_city:
            event_city = event.get('city_name', '').lower().strip()
            
            # Use the same city variations logic
            city_variations = {
                'gurugram': ['gurugram', 'gurgaon'],
                'gurgaon': ['gurugram', 'gurgaon'],
                'delhi': ['delhi', 'new delhi'],
                'new delhi': ['delhi', 'new delhi'],
                'noida': ['noida'],
                'mumbai': ['mumbai', 'bombay'],
                'bangalore': ['bangalore', 'bengaluru'],
                'bengaluru': ['bangalore', 'bengaluru']
            }
            
            current_city_variations = city_variations.get(current_city, [current_city])
            
            # Check if event city matches any variation
            for variation in current_city_variations:
                if variation in event_city or event_city in variation:
                    score += 0.35
                    break
            
            # Also check direct match
            if score == 0 and (current_city in event_city or event_city in current_city):
                score += 0.35
        
        # Activity match (30% weight)
        user_activities = preferences.get('activities', [])
        event_activity = event.get('activity', '').lower()
        if user_activities:
            for activity in user_activities:
                if activity.lower() in event_activity or event_activity in activity.lower():
                    score += 0.30
                    break
        
        # Area match (15% weight)
        user_areas = [area.lower() for area in preferences.get('areas', [])]
        event_area = event.get('area_name', '').lower()
        if user_areas and event_area in user_areas:
            score += 0.15
        
        # Budget match (15% weight)
        user_budget = preferences.get('budget_max')
        event_price = event.get('ticket_price', 0)
        if user_budget is not None:
            if event_price <= user_budget:
                score += 0.15
            elif event_price <= user_budget * 1.2:
                score += 0.08
        
        # Availability (5% weight)
        if event.get('available_spots', 0) > 0:
            score += 0.05
        
        return min(score, 1.0)

    def _generate_recommendation_reason_enhanced(self, event: dict, preferences: dict, score: float, current_city: str) -> str:
        """Generate reason with city emphasis"""
        reasons = []
        
        # City match (most important)
        event_city = event.get('city_name', '')
        if current_city and (current_city.lower() in event_city.lower() or event_city.lower() in current_city.lower()):
            reasons.append(f"in your current city ({event_city})")
        
        # Activity match
        user_activities = preferences.get('activities', [])
        event_activity = event.get('activity', '')
        if user_activities and any(act.lower() in event_activity.lower() for act in user_activities):
            reasons.append(f"matches your interest in {event_activity}")
        
        # Area preference
        event_area = event.get('area_name', '')
        user_areas = preferences.get('areas', [])
        if user_areas and event_area.lower() in [a.lower() for a in user_areas]:
            reasons.append(f"in preferred area {event_area}")
        
        # Score-based
        if score > 0.8:
            reasons.append("perfect match for you")
        elif score > 0.6:
            reasons.append("highly recommended")
        
        return ', '.join(reasons).capitalize() if reasons else "Recommended based on your preferences"

    def _generate_natural_response(self, events: list, query: str, current_city: str) -> str:
        """Generate natural language response for UI"""
        if not events:
            return f"I couldn't find any events matching your request in {current_city}. Try adjusting your preferences or checking other cities."
        
        response = f"Great news! I found {len(events)} amazing events for you"
        if current_city:
            response += f" in {current_city.title()}"
        response += ":\n\n"
        
        for i, event in enumerate(events[:3], 1):  # Show top 3
            response += f"{i}. 🎉 **{event['name']}**\n"
            response += f"   📍 {event['loc`ation']['venue']}, {event['location']['area']}\n"
            response += f"   ⏰ {event['start_time']} to {event['end_time']}\n"
            response += f"   💰 ₹{event['price']}\n"
            response += f"   🔗 Register: {event['registration_url']}\n\n"
        
        if len(events) > 3:
            response += f"...and {len(events) - 3} more great options!"
        
        return response

    def get_bot_response_json(self, user_message: str, user_id: str = None) -> str:
        """Get bot response in JSON-friendly format"""
        try:
            # Similar to get_bot_response but returns just the text
            response = self.get_bot_response(user_message, user_id)
            # Clean up any console formatting
            return response.replace("Miffy: ", "").strip()
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

# Create bot instance
bot = MeetupBot()

# Instructions for use
print("🚀 Welcome to Miffy - Your Personal Meetup Recommendation Assistant!")
print("=" * 60)
print("📋 Instructions:")
print("1. Miffy will automatically sync events from the API")
print("2. Start chatting with Miffy about events you're interested in")
print("3. Miffy will recommend events based on your preferences")
print("4. Type 'quit' to exit the conversation")
print("")
print("🔥 NEW: JSON API for Recommendations!")
print("• bot.get_recommendations_json(request_data)")
print("• Returns structured JSON with match scores and reasons")
print("• Perfect for mobile apps and web integrations!")
print("=" * 60)

# Step 1: Check if data already exists
print("\n📁 Step 1: Checking for existing data...")
print(f"🔧 Debug: ChromaDB server: {bot.chroma_manager.host}:{bot.chroma_manager.port}")

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
    print(f"✅ Found {existing_events} events in ChromaDB! Using existing data.")
    print("🔄 Running incremental sync to get latest updates...")
    bot.sync_events_once(full_sync=False)  # Incremental sync to get updates
else:
    print("❌ No existing events found. Running full initial sync...")
    bot.sync_events_once(full_sync=True)  # Full sync for first time
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
        print("ℹ️ No user preferences found.")
        # Diagnostics
        bot.chroma_manager.validate_user_prefs_setup()
    else:
        print(f"✅ Found {existing_user_prefs} existing user preferences in ChromaDB!")
    
    # Always offer CSV import option
    print("\n📥 CSV Import Options:")
    print("You can import user preferences from CSV file:")
    print("- Upload a new CSV file")
    print("- Replace existing preferences")
    print("- Add to existing preferences")
    
    try:
        tmp_mgr = bot.user_pref_sync_manager
        imported = tmp_mgr.prompt_and_import_csv_interactive()
        if imported > 0:
            print(f"✅ Imported {imported} user preferences from CSV", flush=True)
            user_prefs_info = bot.chroma_manager.get_user_prefs_stats()
            existing_user_prefs = user_prefs_info.get('total_user_preferences', 0)
            print(f"🔧 Debug: Total user preferences count after CSV import: {existing_user_prefs}")
        else:
            print("ℹ️ No CSV import performed.")
    except Exception as e:
        print(f"❌ CSV import attempt failed: {e}", flush=True)
except Exception as e:
    print(f"❌ Error checking user preferences: {e}")

if existing_events > 0:
    print("\n🎉 Great! Event data ready.")
    # Step 2: Start conversation
    print("\n💬 Step 2: Start chatting with Miffy")
    print("Try asking things like:")
    print("- 'Looking for something fun to do today'")
    print("- 'I want to play football in Mumbai'")
    print("- 'Show me tech events under ₹500'")
    print("- 'I enjoy hiking and outdoor activities on weekends'")
    print("\n🎯 **For best results**: Share your preferences first!")
    print("   The Miffy will ask about your interests, location, and budget if not provided.")

    # Updated synchronization options
    print("\n🔄 Event Synchronization Options:")
    print("1. bot.start_conversation() - Start with existing data (recommended)")
    print("2. bot.start_with_sync(2) - Start with 2-minute incremental sync")
    print("3. bot.sync_events_once(full_sync=False) - Run incremental sync")
    print("4. bot.sync_events_once(full_sync=True) - Run full sync (clears existing)")
    print("\nNow preserves existing event data and uses incremental updates!")
    print("Periodic sync focuses on '/updated' API for efficient updates every 2 minutes.")
    
    print("\n📋 User Preferences CSV Import Methods:")
    print("• bot.user_pref_sync_manager.import_user_preferences_from_csv_path('path/to/file.csv')")
    print("• bot.user_pref_sync_manager.prompt_and_import_csv_interactive()") 
    print("• bot.user_pref_sync_manager.import_user_preferences_from_dataframe(df)")
    
    print("\n🔥 JSON API Usage Examples:")
    print("=" * 50)
    print("# Example 1: Get recommendations for user with saved preferences")
    print('request = {"user_id": ""451 "limit": 3}')
    print('result = bot.get_recommendations_json(request)')
    print()
    print("# Example 2: Override preferences for specific request")
    print('request = {')
    print('    "user_id": "451",')
    print('    "preferences": {')
    print('        "activities": ["badminton", "yoga"],')
    print('        "location": "Mumbai",')
    print('        "budget_max": 500')
    print('    },')
    print('    "limit": 5')
    print('}')
    print('result = bot.get_recommendations_json(request)')
    print()
    print("# Example 3: Search without user_id")
    print('request = {')
    print('    "preferences": {')
    print('        "activities": ["cricket"],')
    print('        "areas": ["Bandra", "Andheri"]')
    print('    },')
    print('    "query": "cricket weekend"')
    print('}')
    print('result = bot.get_recommendations_json(request)')
    print()
    print("# Response includes:")
    print("• match_score (0.0-1.0) for each event")
    print("• why_recommended explanation")
    print("• Complete event details with IST times")
    print("• location details (venue, area, city)")
    print("• Direct registration URLs")
    print("=" * 50)

    bot.start_conversation()
else:
    print("❌ Please check the API connection and try again.")