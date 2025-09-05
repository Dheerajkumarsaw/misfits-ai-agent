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

    def _initialize_collection(self, force_recreate: bool = True):
        """Initialize the events collection with embedding function
        
        Args:
            force_recreate: If True, delete and recreate the collection (default: True)
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

    def reinitialize_events_collection(self):
        """Reinitialize the events collection, clearing all existing data"""
        print("🔄 Reinitializing events collection...")
        self.collection = self._initialize_collection(force_recreate=True)
        print("✅ Events collection reinitialized")
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
                # Try multiple URL field names
                event_url = (
                    event.get('registration_url') or 
                    event.get('event_url') or 
                    event.get('signup_url') or 
                    event.get('booking_url') or 
                    event.get('link') or 
                    event.get('url') or 'N/A'
                )
                events_context += f"  URL: {event_url}\n\n"
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
5. If the user provides a user_id, FIRST retrieve their preferences from the user_preferences collection and tailor recommendations accordingly. If there is no user preference data avilable first ask the user to share their preferences. ask what are thier hobbies and intrestes and based on that then recommend them the meetups after the user share their preferences.
6. And user has himself provided the suggestion and preferences then repeat instruction no 1,2 and 3 based on his input.
7. Be concise, friendly, and helpful. Use a few tasteful emojis.
8. Never invent facts. If a field is missing, say “N/A”.
9. Prefer recent/upcoming events over past ones.
10. Always retrive the data of events from the vector database only with consise information about the all the fields. And the registration url should be perfectally fetched from the database and exact. 

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
    
    print("\n📋 User Preferences CSV Import Methods:")
    print("• bot.user_pref_sync_manager.import_user_preferences_from_csv_path('path/to/file.csv')")
    print("• bot.user_pref_sync_manager.prompt_and_import_csv_interactive()") 
    print("• bot.user_pref_sync_manager.import_user_preferences_from_dataframe(df)")

    bot.start_conversation()
else:
    print("❌ Please check the API connection and try again.")