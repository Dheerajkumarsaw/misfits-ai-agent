# Interactive Meetup Recommendation Bot with ChromaDB Integration
# Install required packages at the start
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
import httpx
from concurrent.futures import ThreadPoolExecutor

# Initialize the NVIDIA API client
http_client = httpx.Client(timeout=30.0)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
    http_client=http_client
)

# EventDetailsForAgent class to match the gRPC message structure
class EventDetailsForAgent:
    def __init__(self, event_id=None, event_name="", description="", activity="",
                 start_time=None, end_time=None, allowed_friends=0, ticket_price=0,
                 event_url="", available_spots=0, location_name="", location_url="",
                 area_name="", city_name="", club_name="", payment_terms="",
                 activity_icon_url="", club_icon_url="", event_cover_image_url="",
                 event_uuid="", participants_count=0):
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
        # New fields from API
        self.activity_icon_url = activity_icon_url
        self.club_icon_url = club_icon_url
        self.event_cover_image_url = event_cover_image_url
        self.event_uuid = event_uuid
        self.participants_count = participants_count

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
        except Exception:
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
        except Exception:
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
            'payment_terms': self.payment_terms,
            # New fields from API
            'activity_icon_url': self.activity_icon_url,
            'club_icon_url': self.club_icon_url,
            'event_cover_image_url': self.event_cover_image_url,
            'event_uuid': self.event_uuid,
            'participants_count': self.participants_count
        }

class ChromaDBManager:
    def __init__(self, host: str = "43.205.192.16", port: int = 8000):
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
            ('Participants', event_data.get('participants_count')),
            ('Payment', event_data.get('payment_terms')),
            ('Event Link', event_data.get('event_url')),  # Primary event access URL
            ('Location Link', event_data.get('location_url')),
            ('Cover Image', event_data.get('event_cover_image_url')),
            ('Event UUID', event_data.get('event_uuid'))
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
            except Exception:
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
                  # Get count before deletion
                  initial_items = self.collection.get()
                  ids_to_delete = initial_items.get("ids", [])
                  initial_count = len(ids_to_delete)

                  print(f"📊 ChromaDB before deletion: {initial_count} events")

                  if ids_to_delete:
                     batch_size = 100
                     for i in range(0, len(ids_to_delete), batch_size):
                           batch_ids = ids_to_delete[i:i + batch_size]
                           self.collection.delete(ids=batch_ids)
                     print(f"🗑️ Deleted {len(ids_to_delete)} old events")

                     # Wait for ChromaDB to process deletions
                     print("⏳ Waiting for ChromaDB to process deletions...")
                     time.sleep(2)

                     # Verify collection is empty after deletion
                     after_delete = self.collection.get()
                     remaining_count = len(after_delete.get("ids", []))

                     if remaining_count == 0:
                        print(f"✅ Deletion verified: Collection is now empty")
                     else:
                        print(f"⚠️ Warning: Expected 0 events after deletion, found {remaining_count}")
                        print(f"⚠️ Deletion may be incomplete - this could cause data skip issues")
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
               
               # Parse start_time to create numeric timestamp for filtering
               start_time_str = str(event.get('start_time', ''))
               start_timestamp = 0  # Default to 0 if parsing fails

               if start_time_str:
                   try:
                       from datetime import datetime
                       # Parse datetime string (handle both "2025-09-10 15:00" and "2025-09-10 15:00 IST")
                       try:
                           event_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M IST")
                       except ValueError:
                           event_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
                       # Convert to Unix timestamp (float)
                       start_timestamp = event_dt.timestamp()
                   except Exception as e:
                       print(f"⚠️ Could not parse start_time '{start_time_str}' for event {event_id}: {e}")

               metadata = {
                  'event_id': event_id,
                  'name': str(event.get('event_name', '')),
                  'description': self._clean_description(event.get('description')),
                  'activity': str(event.get('activity', '')),
                  'start_time': start_time_str,
                  'start_timestamp': start_timestamp,  # Numeric timestamp for filtering
                  'end_time': str(event.get('end_time', '')),
                  'ticket_price': str(event.get('ticket_price', '')),
                  'event_url': str(event_url),
                  'registration_url': str(event_url),
                  'available_spots': str(event.get('available_spots', 0)),
                  'location_name': str(event.get('location_name', '')),
                  'location_url': str(event.get('location_url', '')),
                  'area_name': str(event.get('area_name', '')),
                  'city_name': str(event.get('city_name', '')),
                  'club_name': str(event.get('club_name', '')),
                  'payment_terms': str(event.get('payment_terms', '')),
                  # New fields from API
                  'activity_icon_url': str(event.get('activity_icon_url', '')),
                  'club_icon_url': str(event.get('club_icon_url', '')),
                  'event_cover_image_url': str(event.get('event_cover_image_url', '')),
                  'event_uuid': str(event.get('event_uuid', '')),
                  'participants_count': str(event.get('participants_count', 0))
               }

               doc_text = self.prepare_event_text(metadata)
               
               documents.append(doc_text)
               metadatas.append(metadata)
               ids.append(event_id)  # Use the converted string ID

         # Add in batches with error handling
         batch_size = 100
         total_batches = (len(documents) + batch_size - 1) // batch_size

         # Track counts for logging and failure detection
         existing_ids = set()
         failed_batches = 0

         if not clear_existing:
            # Get existing IDs to differentiate new vs updated
            try:
                existing_data = self.collection.get(ids=ids, include=[])
                existing_ids = set(existing_data.get('ids', []))
            except Exception as e:
                print(f"⚠️ Could not fetch existing IDs: {e}")

         for i in range(0, len(documents), batch_size):
               try:
                  batch_num = i//batch_size + 1
                  batch_docs = documents[i:i+batch_size]
                  batch_meta = metadatas[i:i+batch_size]
                  batch_ids = ids[i:i+batch_size]

                  # Always use upsert (updates existing + adds new) - safer than add()
                  new_count = sum(1 for bid in batch_ids if bid not in existing_ids)
                  update_count = len(batch_ids) - new_count
                  print(f"📦 Upserting batch {batch_num}/{total_batches}: {new_count} new, {update_count} updates")

                  self.collection.upsert(
                     documents=batch_docs,
                     metadatas=batch_meta,
                     ids=batch_ids
                  )
                  print(f"✅ Upserted batch {batch_num}/{total_batches}")

               except Exception as e:
                  print(f"❌ Failed to upsert batch {batch_num}/{total_batches}: {str(e)}")
                  failed_batches += 1
                  # Try to continue with remaining batches
                  continue

         # Check if any batches failed
         if failed_batches > 0:
            print(f"⚠️ Warning: {failed_batches}/{total_batches} batches failed!")
            return False

         # Wait for ChromaDB to commit the data
         print("⏳ Waiting for ChromaDB to commit data...")
         time.sleep(1)

         # Verify data was actually committed by checking total collection count
         try:
            after_add = self.collection.get()
            actual_count = len(after_add.get("ids", []))

            if clear_existing:
               # For full sync, count should match exactly what we added
               expected_count = len(documents)
               print(f"📊 ChromaDB after addition: {actual_count} events")

               if actual_count == expected_count:
                  print(f"✅ Verification passed: {actual_count} events confirmed in ChromaDB")
               elif actual_count == 0:
                  print(f"❌ Verification FAILED: Expected {expected_count} events, but ChromaDB is EMPTY!")
                  print(f"❌ Data from upcoming API was NOT successfully added")
                  return False
               else:
                  print(f"⚠️ Verification warning: Expected {expected_count}, found {actual_count} events")
            else:
               # For incremental sync, count should increase by new events
               initial_count = len(existing_ids)
               expected_minimum = initial_count + sum(1 for eid in ids if eid not in existing_ids)
               print(f"📊 ChromaDB after upsert: {actual_count} events (was {initial_count})")

               if actual_count >= expected_minimum:
                  added = actual_count - initial_count
                  print(f"✅ Verification passed: Added {added} new events")
               else:
                  print(f"⚠️ Verification warning: Expected at least {expected_minimum}, found {actual_count}")

         except Exception as e:
            print(f"⚠️ Could not verify data commit: {e}")

         # Final summary
         if clear_existing:
            print(f"✅ Successfully added {len(documents)} events to ChromaDB (full sync)")
         else:
            new_total = sum(1 for eid in ids if eid not in existing_ids)
            update_total = len(ids) - new_total
            print(f"✅ Successfully upserted {len(documents)} events: {new_total} new, {update_total} updated")

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
            filters: Dictionary of filters to apply (supports operators like $gte, $gt, $lte, $lt, $eq)

        Returns:
            List of event dictionaries with metadata
        """
        try:
            params = {
                "query_texts": [query],
                "n_results": n_results
            }
            if filters:
                # Build where clause supporting comparison operators
                where = {}
                for key, value in filters.items():
                    if isinstance(value, dict):
                        # Already has operator (e.g., {"$gte": "2025-12-01"})
                        where[key] = value
                    else:
                        # Simple equality
                        where[key] = {"$eq": value}

                if where:
                    params["where"] = where

            results = self.collection.query(**params)

            # Combine metadata with distances for ranking
            events = []
            if results.get('metadatas'):
                for i, metadata in enumerate(results['metadatas'][0]):
                    if metadata:
                        event = metadata.copy()
                        # Add defaults for new fields if missing (backward compatibility)
                        event.setdefault('activity_icon_url', '')
                        event.setdefault('club_icon_url', '')
                        event.setdefault('event_cover_image_url', '')
                        event.setdefault('event_uuid', '')
                        event.setdefault('participants_count', '0')
                        event.setdefault('available_spots', '0')

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
            if results['metadatas']:
                event = results['metadatas'][0]
                # Add defaults for new fields if missing (backward compatibility)
                event.setdefault('activity_icon_url', '')
                event.setdefault('club_icon_url', '')
                event.setdefault('event_cover_image_url', '')
                event.setdefault('event_uuid', '')
                event.setdefault('participants_count', '0')
                event.setdefault('available_spots', '0')
                return event
            return None
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
            f"👥 **Joined**: {event.get('participants_count', 0)}",
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
                        payment_terms=event_data.get('payment_terms', ''),
                        # New fields from API
                        activity_icon_url=event_data.get('activity_icon_url', ''),
                        club_icon_url=event_data.get('club_icon_url', ''),
                        event_cover_image_url=event_data.get('event_cover_image_url', ''),
                        event_uuid=event_data.get('event_uuid', ''),
                        participants_count=event_data.get('participants_count', 0)
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
            from datetime import datetime
            print(f"🔄 Calling Updated events API: {self.updated_api_url}")
            print(f"⏰ Timestamp: {datetime.now().isoformat()}")

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

            print(f"📊 Response Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                events = []

                # Handle both direct list and nested format
                events_data = data.get('updated_and_new_events', []) if isinstance(data, dict) else data
                print(f"📊 Found {len(events_data) if isinstance(events_data, list) else 0} events in response")

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
                        payment_terms=event_data.get('payment_terms', ''),
                        # New fields from API
                        activity_icon_url=event_data.get('activity_icon_url', ''),
                        club_icon_url=event_data.get('club_icon_url', ''),
                        event_cover_image_url=event_data.get('event_cover_image_url', ''),
                        event_uuid=event_data.get('event_uuid', ''),
                        participants_count=event_data.get('participants_count', 0)
                    )
                    events.append(event)

                print(f"✅ Successfully fetched {len(events)} updated events")
                return events
            else:
                print(f"❌ API call failed with status code: {response.status_code}")
                print(f"❌ Response: {response.text[:200]}")  # Only first 200 chars
                return []

        except requests.exceptions.Timeout as e:
            print(f"❌ Timeout error: {e}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return []
        except Exception as e:
            print(f"❌ Error calling updated events API: {e}")
            return []

    def run_single_sync(self, full_sync: bool = False) -> int:
        """Run a single synchronization of events from API

        Args:
            full_sync: If True, calls both /upcoming + /updated APIs (for initial load)
                      If False, only calls /updated API (for incremental updates)

        Returns:
            Number of events attempted to sync (0 if failed)

        API Call Behavior:
            • full_sync=True  → /upcoming API + /updated API (complete refresh)
            • full_sync=False → /updated API only (incremental sync)
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

                # For full sync, upsert all events (no clearing needed)
                if unique_events:
                    event_count = len(unique_events)
                    success = self.chroma_manager.add_events_batch(unique_events, clear_existing=False)
                    if success:
                        print(f"✅ Successfully synchronized {event_count} events (full sync)")
                        # Wait for ChromaDB to fully commit all batches
                        print("⏳ Waiting for ChromaDB to complete batch processing...")
                        time.sleep(2)
                        print("✅ ChromaDB sync confirmed")
                        return event_count
                    else:
                        print("❌ Failed to add events to ChromaDB")
                        return 0
                else:
                    print("ℹ️ No events found in API responses")
                    return 0
            else:
                print("🔄 Running incremental event synchronization...")
                # Only get updated events for incremental sync
                updated_events = self.call_updated_events_api()
                updated_dicts = [e.to_dict() for e in updated_events]

                # Add new/updated events without clearing existing
                if updated_dicts:
                    event_count = len(updated_dicts)
                    success = self.chroma_manager.add_events_batch(updated_dicts, clear_existing=False)
                    if success:
                        print(f"✅ Successfully synchronized {event_count} updated events")
                        # Wait for ChromaDB to fully commit all batches
                        print("⏳ Waiting for ChromaDB to complete batch processing...")
                        time.sleep(2)
                        print("✅ ChromaDB sync confirmed")
                        return event_count
                    else:
                        print("❌ Failed to add updated events to ChromaDB")
                        return 0
                else:
                    print("ℹ️ No updated events found")
                    return 0

        except Exception as e:
            print(f"❌ Error during synchronization: {e}")
            return 0

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
            # Use incremental sync (only /updated API) for periodic updates
            # This is efficient - only calls /updated API to get new/changed events
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

   #  def prompt_and_import_csv_interactive(self) -> int:
   #      """Prompt user to upload or provide path to CSV and import user preferences."""
   #      try:
   #          # Prefer Colab's upload UI if available
   #          # try:

   #          #     print("📤 Please upload a CSV file with user preferences...", flush=True)
   #          #     uploaded = colab_files.upload()
   #          #     if uploaded:
   #          #         name = next(iter(uploaded.keys()))
   #          #         data = uploaded[name]
   #          #         df = pd.read_csv(io.BytesIO(data))
   #          #         count = self.import_user_preferences_from_dataframe(df)
   #          #         print(f"✅ Imported {count} user preferences from uploaded CSV", flush=True)
   #          #         return count
   #          # except Exception:
   #          #     # Not in Colab or upload failed; fall back to path prompt
   #          #     pass

   #          try:
   #              path = input("Enter CSV path to import user preferences (or press Enter to skip): ").strip()
   #          except Exception:
   #              path = ""
   #          if not path:
   #              print("ℹ️ CSV import skipped", flush=True)
   #              return 0
   #          count = self.import_user_preferences_from_csv_path(path)
   #          if count > 0:
   #              print(f"✅ Imported {count} user preferences from {path}", flush=True)
   #          return count
   #      except Exception as e:
   #          print(f"❌ Error during interactive CSV import: {e}", flush=True)
   #          return 0

class UserPreferenceAPISyncManager:
    """API-based user preference sync manager with scheduled daily sync at 12 AM IST"""

    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager
        self.user_api_url = "https://notify.misfits.net.in/api/user/ai-agent/all"
        self.batch_size = 2000
        self.is_running = False
        self.sync_thread = None
        self.last_sync_time = None
        self.last_sync_count = 0

    def fetch_user_preferences_page(self, offset: int = 0, limit: int = 2000) -> tuple:
        """Fetch a page of user preferences from API with ID-based pagination (offset).

        Returns:
            tuple: (list of user dicts, next_offset or None)
        """
        try:
            params = {
                'offset': offset,
                'page_limit': limit
            }

            print(f"📥 Fetching users from API (offset={offset}, page_limit={limit})...", flush=True)
            response = requests.get(self.user_api_url, params=params, timeout=30)

            if response.status_code == 404:
                print("⚠️ User API endpoint not found. User sync will be skipped.", flush=True)
                return [], None

            response.raise_for_status()
            data = response.json()

            users = data.get('users', [])

            # Extract highest user_id as next offset
            next_offset = None
            if users:
                user_ids = [int(u.get('user_id') or u.get('id', 0)) for u in users if u.get('user_id') or u.get('id')]
                next_offset = max(user_ids) if user_ids else None

            print(f"✅ Fetched {len(users)} users. Next offset: {next_offset}", flush=True)
            return users, next_offset

        except Exception as e:
            print(f"❌ Error fetching user preferences page: {e}", flush=True)
            return [], None

    def run_full_sync(self) -> int:
        """Run full user preference sync with pagination and upsert.

        Returns:
            Number of users synced
        """
        try:
            print("🔄 Starting API-based user preference sync...", flush=True)

            total_synced = 0
            offset = 0

            while True:
                # Fetch page
                users, next_offset = self.fetch_user_preferences_page(offset, self.batch_size)

                if not users:
                    break

                # Convert to DataFrame format for existing import method
                df = pd.DataFrame(users)

                # Get the UserPreferenceSyncManager to use its import method
                temp_manager = UserPreferenceSyncManager(self.chroma_manager)
                synced_count = temp_manager.import_user_preferences_from_dataframe(df)
                total_synced += synced_count

                print(f"✅ Synced batch of {synced_count} users (total: {total_synced})", flush=True)

                # Check if there are more pages
                if next_offset is None:
                    break

                offset = next_offset

            self.last_sync_time = datetime.now()
            self.last_sync_count = total_synced
            print(f"✅ User preference sync completed. Total synced: {total_synced}", flush=True)
            return total_synced

        except Exception as e:
            print(f"❌ Error during user preference sync: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 0

    def get_next_sync_time_ist(self) -> datetime:
        """Calculate next sync time at 12:00 AM IST"""
        # IST is UTC+5:30
        ist_offset = timedelta(hours=5, minutes=30)
        now_utc = datetime.now(timezone.utc)
        now_ist = now_utc + ist_offset

        # Calculate next midnight (12:00 AM IST)
        next_sync_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        return next_sync_ist

    def start_scheduled_sync(self):
        """Start scheduled user preference sync at 12:00 AM IST daily"""
        if self.is_running:
            print("⚠️ User preference API sync is already running", flush=True)
            return

        self.is_running = True

        def sync_loop():
            # Run initial sync immediately
            print(f"🔄 Running initial user preference sync...", flush=True)
            self.run_full_sync()

            # Schedule daily sync at 12 AM IST
            while self.is_running:
                try:
                    # Calculate time until next 12 AM IST
                    ist_offset = timedelta(hours=5, minutes=30)
                    now_utc = datetime.now(timezone.utc)
                    now_ist = now_utc + ist_offset

                    # Next 12 AM IST
                    next_sync = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

                    time_until_sync = (next_sync - now_ist).total_seconds()

                    print(f"⏰ Next user sync scheduled at 12:00 AM IST ({next_sync.strftime('%Y-%m-%d %H:%M:%S')} IST)", flush=True)
                    print(f"⏰ Time until next sync: {time_until_sync / 3600:.1f} hours", flush=True)

                    # Sleep until next sync time (check every minute for stop signal)
                    elapsed = 0
                    while elapsed < time_until_sync and self.is_running:
                        sleep_time = min(60, time_until_sync - elapsed)  # Sleep in 1-minute chunks
                        time.sleep(sleep_time)
                        elapsed += sleep_time

                    # Run sync if still running
                    if self.is_running:
                        print(f"🔄 Running scheduled user preference sync (12:00 AM IST)...", flush=True)
                        self.run_full_sync()

                except Exception as e:
                    print(f"❌ Error in scheduled sync loop: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    # Sleep 1 hour before retrying
                    time.sleep(3600)

        self.sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self.sync_thread.start()
        print(f"✅ Started user preference scheduled sync (daily at 12:00 AM IST)", flush=True)

    def stop_scheduled_sync(self):
        """Stop scheduled synchronization"""
        self.is_running = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)
        print("🛑 Stopped user preference scheduled sync")

    def get_sync_status(self) -> dict:
        """Get current sync status"""
        return {
            'is_running': self.is_running,
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'last_sync_count': self.last_sync_count,
            'next_sync_time': self.get_next_sync_time_ist().isoformat() if self.is_running else None
        }

# ============================================================================
# PERFORMANCE CACHING SYSTEM
# ============================================================================

class PerformanceCacheManager:
    """
    Multi-tier in-memory cache for AI agent performance optimization

    Cache Tiers:
    - Tier 1: Query Analysis (LLM results) - Expensive, shared across users
    - Tier 2: User Context (preferences) - Per-user, medium cost
    - Tier 3: Vector Search (ChromaDB results) - Moderate cost, query-specific
    - Tier 4: Final Results (complete responses) - Full response caching

    Uses cachetools.TTLCache for automatic expiration
    """

    def __init__(self):
        from cachetools import TTLCache

        # Tier 1: Query Analysis Cache (user-independent, saves LLM calls)
        # Key format: "qanalysis:{query_hash}:{pref_context_hash}"
        self.query_analysis_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour, 1000 queries

        # Tier 2: User Context Cache (per-user preferences and metadata)
        # Key format: "user_prefs:{user_id}"
        self.user_context_cache = TTLCache(maxsize=10000, ttl=1800)  # 30 min, 10k users

        # Tier 3: Vector Search Cache (ChromaDB search results)
        # Key format: "vsearch:{search_query_hash}:{filters_hash}"
        self.vector_search_cache = TTLCache(maxsize=2000, ttl=300)  # 5 min, 2000 searches

        # Tier 4: Final Results Cache (complete recommendation responses)
        # Key format: "results:{user_id}:{query_hash}:{city}"
        self.results_cache = TTLCache(maxsize=3000, ttl=300)  # 5 min, 3000 results

        # Cache statistics tracking
        self.cache_stats = {
            'hits': {
                'query_analysis': 0,
                'user_context': 0,
                'vector_search': 0,
                'results': 0
            },
            'misses': {
                'query_analysis': 0,
                'user_context': 0,
                'vector_search': 0,
                'results': 0
            },
            'total_time_saved_ms': 0,
            'start_time': datetime.now()
        }

    def generate_cache_key(self, *args) -> str:
        """
        Generate stable, deterministic hash key from arguments

        Args:
            *args: Any number of arguments to hash together

        Returns:
            16-character MD5 hex hash
        """
        import hashlib
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def get_cache_stats(self) -> dict:
        """
        Get comprehensive cache performance statistics

        Returns:
            Dictionary with hit rates, time saved, and per-tier metrics
        """
        total_hits = sum(self.cache_stats['hits'].values())
        total_misses = sum(self.cache_stats['misses'].values())
        total_requests = total_hits + total_misses

        uptime = (datetime.now() - self.cache_stats['start_time']).total_seconds()

        return {
            'overall': {
                'hit_rate_percent': round((total_hits / total_requests * 100), 1) if total_requests > 0 else 0,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_requests': total_requests,
                'time_saved_seconds': round(self.cache_stats['total_time_saved_ms'] / 1000, 2),
                'uptime_seconds': round(uptime, 0)
            },
            'by_tier': {
                'query_analysis': {
                    'hits': self.cache_stats['hits']['query_analysis'],
                    'misses': self.cache_stats['misses']['query_analysis'],
                    'hit_rate': self._calculate_hit_rate('query_analysis')
                },
                'user_context': {
                    'hits': self.cache_stats['hits']['user_context'],
                    'misses': self.cache_stats['misses']['user_context'],
                    'hit_rate': self._calculate_hit_rate('user_context')
                },
                'vector_search': {
                    'hits': self.cache_stats['hits']['vector_search'],
                    'misses': self.cache_stats['misses']['vector_search'],
                    'hit_rate': self._calculate_hit_rate('vector_search')
                },
                'results': {
                    'hits': self.cache_stats['hits']['results'],
                    'misses': self.cache_stats['misses']['results'],
                    'hit_rate': self._calculate_hit_rate('results')
                }
            },
            'cache_sizes': {
                'query_analysis': len(self.query_analysis_cache),
                'user_context': len(self.user_context_cache),
                'vector_search': len(self.vector_search_cache),
                'results': len(self.results_cache)
            }
        }

    def _calculate_hit_rate(self, tier: str) -> str:
        """Calculate hit rate percentage for a specific cache tier"""
        hits = self.cache_stats['hits'][tier]
        misses = self.cache_stats['misses'][tier]
        total = hits + misses
        if total == 0:
            return "0.0%"
        return f"{(hits / total * 100):.1f}%"

    def clear_all_caches(self):
        """Clear all cache tiers (useful for testing or manual refresh)"""
        self.query_analysis_cache.clear()
        self.user_context_cache.clear()
        self.vector_search_cache.clear()
        self.results_cache.clear()
        print("🗑️ All caches cleared")

    def clear_user_cache(self, user_id: str):
        """
        Clear all cache entries for a specific user
        Called when user updates their preferences

        Args:
            user_id: User ID to clear cache for
        """
        # Clear user context
        user_pref_key = f"user_prefs:{user_id}"
        if user_pref_key in self.user_context_cache:
            del self.user_context_cache[user_pref_key]

        # Clear results cache for this user (keys start with "results:{user_id}:")
        results_keys_to_clear = [
            k for k in list(self.results_cache.keys())
            if isinstance(k, str) and k.startswith(f"results:{user_id}:")
        ]
        for key in results_keys_to_clear:
            if key in self.results_cache:
                del self.results_cache[key]

        print(f"🗑️ Cleared cache for user {user_id} ({len(results_keys_to_clear) + 1} entries)")

class MeetupBot:
    def __init__(self, auto_sync: bool = True):
        # Initialize performance cache FIRST
        self.cache_manager = PerformanceCacheManager()
        print("✅ Performance caching system initialized")

        self.events_data = None
        self.user_conversations = {}  # Store conversation history per user_id
        self.chroma_manager = ChromaDBManager()
        self.event_sync_manager = EventSyncManager(self.chroma_manager)
        self.user_pref_sync_manager = UserPreferenceSyncManager(self.chroma_manager)
        self.user_api_sync_manager = UserPreferenceAPISyncManager(self.chroma_manager)

        # ============================================================================
        # ACTIVITY SIMILARITY MATCHING - Initialize once at class level
        # ============================================================================
        # Activity category mappings for semantic understanding
        self.activity_categories = {
            'sports': ['football', 'cricket', 'badminton', 'tennis', 'basketball', 'volleyball', 'swimming', 'box_cricket', 'pickleball'],
            'outdoor_sports': ['football', 'cricket', 'tennis', 'hiking', 'trekking', 'cycling', 'running', 'badminton', 'pickleball'],
            'indoor_sports': ['badminton', 'basketball', 'volleyball', 'chess', 'boardgaming', 'bowling'],
            'fitness': ['gym', 'yoga', 'running', 'cycling', 'swimming', 'dance', 'mindfulness'],
            'arts': ['art', 'dance', 'music', 'photography', 'drama', 'theater', 'poetry', 'writing'],
            'creative': ['art', 'photography', 'content_creation', 'writing', 'poetry', 'drama'],
            'tech': ['tech', 'technology', 'coding', 'startup', 'business', 'community_space'],
            'entertainment': ['films', 'movies', 'music', 'pop_culture', 'quiz', 'drama', 'theater'],
            'social': ['boardgaming', 'social_deductions', 'book_club', 'community_space', 'food', 'networking'],
            'wellness': ['yoga', 'mindfulness', 'meditation', 'journaling', 'inner_journey']
        }

        # All available activities for fuzzy matching
        self.all_activities_list = [
            'football', 'cricket', 'badminton', 'tennis', 'swimming', 'gym', 'yoga',
            'dance', 'music', 'art', 'photography', 'hiking', 'trekking', 'cycling',
            'tech', 'coding', 'startup', 'business', 'networking', 'food', 'cooking',
            'boardgaming', 'social_deductions', 'book_club', 'box_cricket', 'films',
            'poetry', 'writing', 'harry_potter', 'pop_culture', 'community_space',
            'content_creation', 'bowling', 'mindfulness', 'pickleball', 'journaling',
            'quiz', 'drama', 'theater', 'basketball', 'volleyball', 'chess', 'gaming', 'reading'
        ]

        print("✅ Activity similarity matching initialized")

        # Start daily user sync at 12 AM IST
        print("🔄 Starting daily user preference sync (12:00 AM IST)...")
        self.user_api_sync_manager.start_scheduled_sync()

        # Auto-sync on initialization (optional, enabled by default)
        if auto_sync:
            print("🔄 Auto-syncing latest upcoming events on init...")
            print("📞 Calling /upcoming + /updated APIs (preserving existing events)")
            try:
                # Use full sync to get all current events, but preserve existing ones
                synced_count = self.sync_events_once(full_sync=True)

                if synced_count > 0:
                    # Verify data was actually committed to ChromaDB
                    print("⏳ Verifying data in ChromaDB...")
                    time.sleep(1)

                    collection_info = self.chroma_manager.get_collection_stats()
                    actual_count = collection_info.get('total_events', 0)

                    if actual_count >= synced_count:
                        print(f"✅ Auto-sync verified! {actual_count} events confirmed in ChromaDB")
                    elif actual_count > 0:
                        print(f"⚠️ Partial sync: Expected {synced_count}, found {actual_count} events in ChromaDB")
                    else:
                        print(f"❌ Sync verification failed: Expected {synced_count} events, but ChromaDB is empty!")
                        print("⚠️ Data from upcoming API was NOT successfully added")
                else:
                    print("⚠️ Auto-sync returned 0 events - check API connection or data availability")

            except Exception as e:
                print(f"⚠️ Auto-sync failed: {e}")
                print("You can manually sync with: bot.sync_events_once(full_sync=True)")

    def find_similar_activities(self, activity: str, max_results: int = 10) -> list:
        """
        Find similar activities using fuzzy matching and category expansion

        Args:
            activity: Activity name to match (e.g., "sports", "outdoor activities", "footbal")
            max_results: Maximum number of similar activities to return

        Returns:
            List of similar activity names (deduplicated)
        """
        from difflib import get_close_matches

        activity_lower = activity.lower().strip()
        similar_activities = []

        # 1. Check if it's a category request (e.g., "sports", "fitness")
        if activity_lower in self.activity_categories:
            # Return all activities in this category
            similar_activities.extend(self.activity_categories[activity_lower])
            print(f"🏷️ Category match: '{activity}' → {len(similar_activities)} activities")
            return similar_activities[:max_results]

        # 2. Check for partial category matches (e.g., "outdoor activities" → "outdoor_sports")
        for category, activities in self.activity_categories.items():
            # Check if query contains category keywords
            category_words = category.replace('_', ' ').lower()
            if category_words in activity_lower or activity_lower in category_words:
                similar_activities.extend(activities)
                print(f"🏷️ Partial category match: '{activity}' matched '{category}' → {len(activities)} activities")

        # 3. Exact match
        if activity_lower in self.all_activities_list:
            if activity_lower not in similar_activities:
                similar_activities.append(activity_lower)
                print(f"✅ Exact match: '{activity}'")

        # 4. Fuzzy matching for typos and variations (e.g., "footbal" → "football")
        fuzzy_matches = get_close_matches(
            activity_lower,
            self.all_activities_list,
            n=min(3, max_results),
            cutoff=0.85  # 85% similarity threshold (only match close typos, not random words)
        )
        for match in fuzzy_matches:
            if match not in similar_activities:
                similar_activities.append(match)
                print(f"🔍 Fuzzy match: '{activity}' → '{match}'")

        # 5. If still no matches, check if activity appears in any category
        if not similar_activities:
            for category, activities in self.activity_categories.items():
                for cat_activity in activities:
                    if activity_lower in cat_activity or cat_activity in activity_lower:
                        if cat_activity not in similar_activities:
                            similar_activities.append(cat_activity)
                            print(f"🔎 Substring match: '{activity}' found in '{cat_activity}'")

        # Remove duplicates while preserving order
        seen = set()
        unique_activities = []
        for act in similar_activities:
            if act not in seen:
                seen.add(act)
                unique_activities.append(act)

        return unique_activities[:max_results]

    def _start_performance_timer(self):
        """Start a new performance timing session"""
        import time
        self._perf_timer = {
            'start_time': time.time(),
            'checkpoints': [],
            'last_checkpoint': time.time()
        }

    def _record_checkpoint(self, checkpoint_name: str):
        """Record a performance timing checkpoint"""
        import time
        if not hasattr(self, '_perf_timer'):
            return

        current_time = time.time()
        elapsed = (current_time - self._perf_timer['last_checkpoint']) * 1000  # ms

        self._perf_timer['checkpoints'].append({
            'name': checkpoint_name,
            'elapsed_ms': round(elapsed, 2),
            'total_ms': round((current_time - self._perf_timer['start_time']) * 1000, 2)
        })
        self._perf_timer['last_checkpoint'] = current_time

    def _get_performance_report(self) -> dict:
        """Get detailed performance timing report"""
        if not hasattr(self, '_perf_timer'):
            return {}

        import time
        total_time = (time.time() - self._perf_timer['start_time']) * 1000  # ms

        return {
            'total_ms': round(total_time, 2),
            'checkpoints': self._perf_timer['checkpoints'],
            'cache_stats': self.cache_manager.get_cache_stats()
        }

    def _safe_float(self, value):
        """Safely convert value to float"""
        try:
            if value is None:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_int(self, value):
        """Safely convert value to int"""
        try:
            if value is None:
                return 0
            return int(value)
        except (ValueError, TypeError):
            return 0

    def _get_user_conversation_history(self, user_id: str = None):
        """Get conversation history for specific user"""
        if not user_id:
            return []
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = []
        return self.user_conversations[user_id]
    
    def _add_to_user_conversation(self, user_id: str, role: str, content: str):
        """Add message to user-specific conversation history"""
        if not user_id:
            return
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = []
        self.user_conversations[user_id].append({"role": role, "content": content})
        
        # Keep only last 10 messages per user to avoid memory bloat
        if len(self.user_conversations[user_id]) > 10:
            self.user_conversations[user_id] = self.user_conversations[user_id][-10:]

    def classify_query_intent(self, query: str, saved_preferences: dict = None) -> dict:
        """
        Classify user query to determine if it's asking for specific activities or generic events

        Args:
            query: User's natural language query
            saved_preferences: User's saved preferences (if any)

        Returns:
            dict with:
                - intent_type: "specific_activity", "generic_request", "preference_question", "casual_chat"
                - confidence: 0.0-1.0
                - override_preferences: bool (True if query should override saved preferences)
        """
        query_lower = query.lower().strip()

        # Define explicit activity request patterns
        specific_patterns = [
            r'\b(show|find|get|looking for|search|want|interested in|need)\s+(me\s+)?(\w+)\s+(event|activity|meetup)',
            r'\b(show|find|get)\s+(me\s+)?some\s+(\w+)',
            r'\bi\s+(want to|would like to)\s+(play|do|try|join)\s+(\w+)',
            r'\b(\w+)\s+events?\b',
            r'\bevents?\s+(for|about|related to)\s+(\w+)',
        ]

        # Check if query has specific activity patterns
        for pattern in specific_patterns:
            if re.search(pattern, query_lower):
                return {
                    "intent_type": "specific_activity",
                    "confidence": 0.9,
                    "override_preferences": True
                }

        # Generic request patterns (use saved preferences)
        generic_patterns = [
            r'\b(find|show|get|recommend|suggest)\s+(me\s+)?(some\s+)?events?\b',
            r'\bevents?\s+for\s+me\b',
            r'\bwhat\s+events?\b',
            r'\bany\s+events?\b',
            r'\bevents?\s+happening\b',
        ]

        for pattern in generic_patterns:
            if re.search(pattern, query_lower) and not any(re.search(sp, query_lower) for sp in specific_patterns):
                return {
                    "intent_type": "generic_request",
                    "confidence": 0.8,
                    "override_preferences": False
                }

        # Question about preferences or capabilities
        question_patterns = [
            r'\bwhat (can you|do you|are you)',
            r'\bhow (do you|does)',
            r'\bwho are you',
            r'\btell me about',
            r'\bwhat\'?s your',
        ]

        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                return {
                    "intent_type": "preference_question",
                    "confidence": 0.85,
                    "override_preferences": False
                }

        # Casual chat (greetings, small talk)
        casual_patterns = [
            r'^\b(hi|hello|hey|greetings|good morning|good evening)\b',
            r'^\b(thanks|thank you|bye|goodbye)\b',
        ]

        for pattern in casual_patterns:
            if re.search(pattern, query_lower):
                return {
                    "intent_type": "casual_chat",
                    "confidence": 0.95,
                    "override_preferences": False
                }

        # Default to generic if no clear pattern
        return {
            "intent_type": "generic_request",
            "confidence": 0.5,
            "override_preferences": False
        }

    def analyze_user_query(self, query: str, saved_preferences: dict = None) -> dict:
        """
        Use LLM to analyze user query and extract intent, activities, and preferences
        NOW WITH INTELLIGENT CACHING - Same query = instant results!

        Args:
            query: User's natural language query
            saved_preferences: User's saved preferences (optional)

        Returns:
            dict with extracted information:
                - activities: List of requested activities
                - location: Requested location (if any)
                - budget: Budget constraints (if any)
                - timing: Time preferences (if any)
                - is_specific_request: Whether user is asking for specific activities
                - should_override_saved: Whether to override saved preferences
        """
        import time
        start_time = time.time()

        try:
            # ============================================================================
            # CACHE CHECK - Avoid expensive LLM call if we've seen this query before
            # ============================================================================
            # Generate cache key based on query + preference context
            pref_context = ""
            if saved_preferences:
                # Include activities in cache key (queries with different saved prefs may have different results)
                activities = saved_preferences.get('activities', [])
                if isinstance(activities, list):
                    pref_context = ",".join(sorted(activities))

            cache_key = f"qanalysis:{self.cache_manager.generate_cache_key(query.lower().strip())}:{self.cache_manager.generate_cache_key(pref_context)}"

            # Check if we've analyzed this query before
            if cache_key in self.cache_manager.query_analysis_cache:
                cached_result = self.cache_manager.query_analysis_cache[cache_key]
                elapsed_ms = (time.time() - start_time) * 1000

                # Update cache statistics
                self.cache_manager.cache_stats['hits']['query_analysis'] += 1
                self.cache_manager.cache_stats['total_time_saved_ms'] += 600  # Avg LLM call time

                print(f"⚡ CACHE HIT: Query analysis from cache ({elapsed_ms:.1f}ms) - Saved ~600ms LLM call")
                return cached_result

            # Cache miss - need to run full analysis
            self.cache_manager.cache_stats['misses']['query_analysis'] += 1
            print(f"💭 Cache miss - Analyzing query with LLM...")

            # First use classification to determine intent
            intent_info = self.classify_query_intent(query, saved_preferences)

            # Define all possible activities for reference
            all_activities = [
                'football', 'cricket', 'badminton', 'tennis', 'swimming', 'gym', 'yoga',
                'dance', 'music', 'art', 'photography', 'hiking', 'trekking', 'cycling',
                'tech', 'coding', 'startup', 'business', 'networking', 'food', 'cooking',
                'boardgaming', 'social_deductions', 'book_club', 'box_cricket', 'films',
                'poetry', 'writing', 'harry_potter', 'pop_culture', 'community_space',
                'content_creation', 'bowling', 'mindfulness', 'pickleball',
                'journaling', 'quiz', 'drama', 'theater', 'improv', 'sports', 'fitness',
                'basketball', 'volleyball', 'chess', 'gaming', 'reading'
            ]

            # Prepare LLM prompt for query analysis
            analysis_prompt = f"""Analyze the following user query and extract structured information.

User Query: "{query}"

{"User's Saved Preferences (for context): " + str(saved_preferences) if saved_preferences else "No saved preferences"}

Available Activities: {', '.join(all_activities)}

Task: Extract the following information from the query in JSON format:
1. activities: List of specific activities the user is asking for (only if explicitly mentioned)
2. location: Any city/area mentioned
3. budget: Any budget/price constraints mentioned
4. timing: Any time preferences (morning/evening/weekend/etc)
5. is_specific_request: true if user is asking for specific activities, false if generic
6. should_override_saved: true if this request should override saved preferences

Examples:
- "show me badminton events" → activities: ["badminton"], is_specific_request: true, should_override_saved: true
- "find events for me" → activities: [], is_specific_request: false, should_override_saved: false
- "looking for tennis in Mumbai" → activities: ["tennis"], location: "Mumbai", is_specific_request: true
- "I want to try cricket" → activities: ["cricket"], is_specific_request: true

Return ONLY a valid JSON object, no other text:"""

            if client is None:
                # Fallback to regex-based extraction if LLM is unavailable
                print("⚠️ LLM unavailable, using fallback extraction")
                fallback_result = self._fallback_query_analysis(query, all_activities, intent_info)
                # Cache fallback result
                self.cache_manager.query_analysis_cache[cache_key] = fallback_result
                return fallback_result

            # Call LLM for analysis
            completion = client.chat.completions.create(
                model="qwen/qwq-32b",
                messages=[
                    {"role": "system", "content": "You are a precise query analyzer. Extract information and return only valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            response_text = completion.choices[0].message.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())

                # Ensure all required fields are present
                result = {
                    "activities": analysis_result.get("activities", []),
                    "location": analysis_result.get("location", ""),
                    "budget": analysis_result.get("budget", ""),
                    "timing": analysis_result.get("timing", ""),
                    "is_specific_request": analysis_result.get("is_specific_request", False),
                    "should_override_saved": analysis_result.get("should_override_saved", False),
                    "confidence": intent_info.get("confidence", 0.7)
                }

                # Store result in cache for future use
                self.cache_manager.query_analysis_cache[cache_key] = result

                elapsed_ms = (time.time() - start_time) * 1000
                print(f"🔍 Query Analysis: {result} (completed in {elapsed_ms:.1f}ms, cached for reuse)")
                return result
            else:
                # If JSON parsing fails, use fallback
                fallback_result = self._fallback_query_analysis(query, all_activities, intent_info)
                # Cache fallback result too
                self.cache_manager.query_analysis_cache[cache_key] = fallback_result
                return fallback_result

        except Exception as e:
            print(f"⚠️ Error in query analysis: {e}")
            # Fallback to regex-based extraction
            fallback_result = self._fallback_query_analysis(query, all_activities, intent_info)
            # Cache fallback result (better than nothing)
            self.cache_manager.query_analysis_cache[cache_key] = fallback_result
            return fallback_result

    def _fallback_query_analysis(self, query: str, all_activities: list, intent_info: dict) -> dict:
        """Fallback query analysis using regex when LLM is unavailable"""
        query_lower = query.lower()

        # Extract activities mentioned in query
        found_activities = []
        for activity in all_activities:
            if re.search(rf'\b{re.escape(activity)}\b', query_lower):
                found_activities.append(activity)

        # Extract location
        location = ""
        location_match = re.search(r'\b(?:in|at|near|around)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b', query, re.IGNORECASE)
        if location_match:
            location = location_match.group(1)

        # Extract budget
        budget = ""
        budget_match = re.search(r'₹?\s*(\d+)(?:\s*-\s*₹?\s*(\d+))?|(?:under|below)\s+₹?\s*(\d+)', query)
        if budget_match:
            budget = budget_match.group(0)

        # Extract timing
        timing = ""
        timing_patterns = ['weekend', 'weekday', 'morning', 'evening', 'afternoon', 'night', 'today', 'tomorrow']
        for time_word in timing_patterns:
            if time_word in query_lower:
                timing = time_word
                break

        return {
            "activities": found_activities,
            "location": location,
            "budget": budget,
            "timing": timing,
            "is_specific_request": len(found_activities) > 0 or intent_info["intent_type"] == "specific_activity",
            "should_override_saved": intent_info.get("override_preferences", len(found_activities) > 0),
            "confidence": 0.7
        }

    def extract_events_from_response(self, response_text: str, fallback_events: list = None) -> list:
        """Extract structured event data from AI response text or use fallback"""
        try:
            # First try to find JSON array in the response
            json_match = re.search(r'\[\s*\{[^\]]+\]', response_text, re.DOTALL)
            if json_match:
                try:
                    events_json = json.loads(json_match.group())
                    if isinstance(events_json, list):
                        return events_json
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction failed, use fallback events if provided
            if fallback_events:
                return fallback_events
            
            # Try to extract event information from natural language
            events = []
            event_pattern = r'(?:Event|\d+\.)\s*([^\n]+?)(?:\n|$)'
            matches = re.findall(event_pattern, response_text)
            
            for match in matches[:5]:  # Limit to 5 events
                # Try to parse event details from text
                event = {
                    "name": match.strip(),
                    "activity": "",
                    "location": {},
                    "price": 0,
                    "registration_url": ""
                }
                events.append(event)
            
            return events if events else fallback_events or []
            
        except Exception as e:
            print(f"Error extracting events: {e}")
            return fallback_events or []

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
            response += f"👥 **Joined**: {event.get('participants_count', 0)}\n"
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

    def prepare_context(self, user_message, user_id: str = None):
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
            # Look for patterns like [User ID: 123] or user_id: 123 or uid 123
            # First check for the format we're using: [User ID: 123]
            id_match = re.search(r"\[User ID:\s*(\d+)\]", user_message)
            if not id_match:
                # Fallback to other patterns
                id_match = re.search(r"(?:user[_ ]?id|uid|id)[:\s]*(\d{3,})", user_message, flags=re.IGNORECASE)
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
        user_history = self._get_user_conversation_history(user_id)
        is_initial_request = len(user_history) == 0
        
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
        for msg in user_history[-6:]:  # Use user-specific history
            history_context += f"{msg['role']}: {msg['content']}\n"

        system_prompt = f"""You are a warm, enthusiastic, and friendly meetup recommendation assistant named Miffy. You love helping people discover exciting events and make new connections. You have access to a vector database of events and provide personalized recommendations with genuine enthusiasm.

{events_context}
{history_context}
{user_pref_context}

Instructions:

RESPONSE FORMAT FOR EVENT RECOMMENDATIONS:
- When providing event recommendations, ALWAYS include a JSON array of events
- Format: After your friendly message, add: "EVENTS_JSON: [{{event1}}, {{event2}}, ...]"
- Each event object must include: event_id, name, club_name, activity, start_time, location (with venue, area, city), price, registration_url
- Example: EVENTS_JSON: [{{"event_id": "123", "name": "Tech Meetup", "club_name": "Tech Club", "activity": "Technology", "start_time": "2024-01-20 18:00", "location": {{"venue": "Hub", "area": "Downtown", "city": "Mumbai"}}, "price": 500, "registration_url": "https://example.com"}}]

CRITICAL USER PREFERENCE MATCHING - NEW INTELLIGENT SYSTEM:
⭐ **SMART QUERY UNDERSTANDING**: The system now intelligently detects user intent and overrides saved preferences when needed

**QUERY PRIORITY RULES** (in order of importance):
1. **EXPLICIT ACTIVITY REQUEST** (Highest Priority)
   - User says "show me badminton events" or "looking for tennis" → Show ONLY that activity
   - Even if user has "cricket" saved preference → Override and show requested activity
   - System analyzes query using LLM to detect explicit activity requests
   - Examples:
     * "find tennis events" + saved cricket → Show TENNIS events
     * "I want to try badminton" + saved football → Show BADMINTON events
     * "show me dance classes" + saved yoga → Show DANCE events

2. **GENERIC EVENT REQUEST** (Use Saved Preferences)
   - User says "find events" or "show me events" or "events for me" → Use SAVED preferences
   - No specific activity mentioned → Use their saved activity preferences
   - Examples:
     * "find events for me" + saved cricket → Show CRICKET events
     * "show me some events" + saved music → Show MUSIC events
     * "what events are available" + saved yoga → Show YOGA events

**IMPLEMENTATION DETAILS**:
- System uses LLM-based query analysis to understand user intent
- Query activities get +50% scoring boost vs saved preferences (+25%)
- This ensures explicitly requested activities always rank higher
- Saved preferences are ONLY used when query is generic (no specific activities mentioned)

**KEY BEHAVIOR**:
- Explicit request ALWAYS overrides saved preferences
- Generic request ALWAYS uses saved preferences
- Better to show 2 relevant requested events than 10 saved preference events
- System is now smart enough to understand variations: "looking for", "interested in", "want to try"

IMPORTANT - ONLY SHOW FUTURE EVENTS:
- NEVER recommend events that have already passed
- Always check event start_time against current date/time
- Filter out any events with start_time in the past
- If an event doesn't have a start_time, be cautious about recommending it
- Today's events are OK if they haven't started yet

PERSONALITY & TONE:
- Be warm, enthusiastic, and conversational (not robotic)
- Use varied greetings and expressions each time
- Show genuine excitement about events you recommend
- Add personality with phrases like "Oh, this looks perfect for you!" or "You're going to love this one!"
- Keep it friendly but not overly casual

CONVERSATION HANDLING:
- Respond naturally to greetings (hi, hello, hey) without forcing event suggestions
- Only suggest events when users explicitly ask or show clear interest
- Keep casual responses brief and welcoming - invite them to explore events
- Let users guide the conversation - don't be pushy about recommendations
- For simple greetings, just be friendly and ask how you can help

INTENT UNDERSTANDING (Be Smart!):
- Analyze user's ACTUAL intent, not just keywords
- "I don't want events" = NEGATIVE intent → Don't search, acknowledge politely
- "Find cricket events" = POSITIVE intent → Search and recommend
- "Restart conversation" = RESET intent → Clear context, start fresh
- "What can you do?" = INFORMATIONAL → Explain your capabilities, don't search
- "Show me activities" = SEARCH intent → Search for activities
- Consider context: "I love events!" (casual talk) vs "Show me events" (search request)

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
        
        # Extract activities/interests - includes all Misfits activities
        activity_keywords = ['football', 'cricket', 'badminton', 'tennis', 'swimming', 'gym', 'yoga', 
                           'dance', 'music', 'art', 'photography', 'hiking', 'trekking', 'cycling',
                           'tech', 'coding', 'startup', 'business', 'networking', 'food', 'cooking',
                           'boardgaming', 'social_deductions', 'book_club', 'box_cricket', 'films', 
                           'poetry', 'writing', 'harry_potter', 'pop_culture', 'community_space', 
                           'content_creation', 'bowling', 'mindfulness', 'others', 'pickleball', 
                           'journaling', 'quiz', 'drama', 'theater', 'improv', 'sports', 'fitness']
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
    
    def generate_personalized_greeting(self, user_id: str = None, user_name: str = None, include_event_teaser: bool = True):
        """Generate a personalized, friendly greeting that varies each time

        Args:
            user_id: Optional user ID to personalize greeting
            user_name: Optional user name to include in greeting
            include_event_teaser: If True, includes event suggestions (default). If False, just friendly greeting.
        """
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
            except Exception:
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

        # If not including event teaser, return simple greeting with CTA
        if not include_event_teaser:
            simple_cta_options = [
                "\n\nHow can I help you today?",
                "\n\nWhat can I help you discover?",
                "\n\nWhat brings you here today?",
                "\n\nHow may I assist you?",
                "\n\nWhat would you like to know about?",
                "\n\nLet me know if you're looking for any events or activities!"
            ]
            return f"{greeting}{random.choice(simple_cta_options)}"

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
    
    def get_bot_response(self, user_message, user_id: str = None):
        """Get response from the AI model with streaming"""
        try:
            # Try to extract user_id from message if not provided
            extracted_user_id = user_id
            if not extracted_user_id:
                try:
                    id_match = re.search(r"(?:user[_ ]?id|uid|id)[:\s]*(\d+)", user_message, re.IGNORECASE)
                    if id_match:
                        extracted_user_id = id_match.group(1)
                    # Also try to extract from [User ID: ...] format
                    bracket_match = re.search(r"\[User ID:\s*(\w+)\]", user_message, re.IGNORECASE)
                    if bracket_match:
                        extracted_user_id = bracket_match.group(1)
                except Exception:
                    pass
            
            # Get user-specific conversation history
            user_history = self._get_user_conversation_history(extracted_user_id)
            
            # Check if this is the first message for this user
            if len(user_history) == 0 and not user_id:  # Only show greeting for CLI mode
                # Try to extract user info from message for greeting
                user_name = None
                is_user_id_request = False
                try:
                    # Check if this is a user ID request
                    if re.search(r"(?:find events?|events?)\s+for\s+user[_ ]?id\s+(\d+)", user_message, re.IGNORECASE):
                        is_user_id_request = True
                    
                    # Try to extract name
                    name_match = re.search(r"(?:i'm|i am|name is|this is)\s+([A-Z][a-z]+)", user_message, re.IGNORECASE)
                    if name_match:
                        user_name = name_match.group(1)
                except Exception:
                    pass
                
                # Only show greeting if it's not a user ID request
                if not is_user_id_request:
                    greeting = self.generate_personalized_greeting(extracted_user_id, user_name)
                    print(f"\nBot: {greeting}\n")
            
            # Add user message to user-specific history
            self._add_to_user_conversation(extracted_user_id, "user", user_message)
            
            # Prepare the context
            system_context = self.prepare_context(user_message, extracted_user_id)
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
            print("📞 This will call /upcoming API to load all current events")
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

    def sync_events_once(self, full_sync: bool = False) -> int:
        """Run a single event synchronization cycle

        Args:
            full_sync: If True, performs full sync (default: False for incremental)

        Returns:
            Number of events synced (0 if failed)
        """
        return self.event_sync_manager.run_single_sync(full_sync=full_sync)

    def sync_user_preferences_once(self, clear_existing: bool = False):
        """Run a full user preferences synchronization cycle using cursor pagination"""
        self.user_pref_sync_manager.run_full_sync(clear_existing=clear_existing)

    def start_user_preferences_daily_sync(self, clear_existing_first_run: bool = False):
        """Start daily user preferences sync (every 24 hours)."""
        self.user_pref_sync_manager.start_periodic_sync(interval_hours=24, clear_existing_first_run=clear_existing_first_run)

    def stop_user_preferences_sync(self):
        """Stop user preferences periodic sync."""
        self.user_pref_sync_manager.stop_periodic_sync()

    def sync_users_once(self) -> int:
        """Run a single user preference synchronization cycle using API

        Returns:
            Number of users synced
        """
        return self.user_api_sync_manager.run_full_sync()

    def get_user_sync_status(self) -> dict:
        """Get current user sync status

        Returns:
            dict with keys: is_running, last_sync_time, last_sync_count, next_sync_time
        """
        return self.user_api_sync_manager.get_sync_status()

    # ============================================================================
    # CACHE MANAGEMENT METHODS
    # ============================================================================

    def invalidate_user_cache(self, user_id: str):
        """
        Invalidate all cache entries for a specific user
        Call this when user updates their preferences to ensure fresh recommendations

        Args:
            user_id: User ID whose cache should be cleared
        """
        self.cache_manager.clear_user_cache(user_id)

    def get_cache_stats(self) -> dict:
        """
        Get comprehensive cache performance statistics

        Returns:
            Dictionary with overall and per-tier cache metrics
        """
        return self.cache_manager.get_cache_stats()

    def clear_all_caches(self):
        """Clear all cache tiers (useful for testing or manual refresh)"""
        self.cache_manager.clear_all_caches()

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
        
        # Budget match removed per user request
        # Budget filtering has been disabled
        
        # Availability bonus (10% weight)
        available_spots = event.get('available_spots', 0)
        # Ensure numeric type
        try:
            available_spots = int(available_spots) if isinstance(available_spots, str) else available_spots
        except (ValueError, TypeError):
            available_spots = 0
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
        
        # Budget match removed per user request
        # Budget filtering has been disabled
        
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
                "preferences": {...},
                "limit": "max number of results (default 5)"
            }
        
        Returns:
            Structured JSON with recommendations filtered by current city
        """
        # Start performance monitoring
        self._start_performance_timer()

        try:
            user_id = request_data.get('user_id')
            user_current_city = request_data.get('user_current_city', '').lower().strip()
            query = request_data.get('query', '')
            override_preferences = request_data.get('preferences', {})
            limit = request_data.get('limit', 50)  # Default to 50 recommendations

            self._record_checkpoint("Request parsing")

            # ============================================================================
            # TIER 4: RESULTS CACHE - Cache complete recommendation responses
            # ============================================================================
            # Generate cache key for final results (user + query + city specific)
            results_cache_key = f"results:{user_id}:{self.cache_manager.generate_cache_key(query.lower().strip())}:{user_current_city}"

            if results_cache_key in self.cache_manager.results_cache:
                # Cache hit - return cached complete response
                cached_response = self.cache_manager.results_cache[results_cache_key]
                self.cache_manager.cache_stats['hits']['results'] += 1
                self.cache_manager.cache_stats['total_time_saved_ms'] += 400  # Entire pipeline
                self._record_checkpoint("Results cache HIT - returning cached response")
                print(f"⚡⚡ TIER 4 CACHE HIT: Complete response from cache - Saved ~400ms (entire pipeline)")
                return cached_response

            # Cache miss - continue with full recommendation pipeline
            self.cache_manager.cache_stats['misses']['results'] += 1
            self._record_checkpoint("Results cache miss - proceeding with full pipeline")

            # Get user preferences (with caching)
            user_preferences = {}
            if user_id:
                # ============================================================================
                # USER PREFERENCE CACHING - Avoid ChromaDB lookup for repeat users
                # ============================================================================
                cache_key = f"user_prefs:{user_id}"

                if cache_key in self.cache_manager.user_context_cache:
                    # Cache hit - use cached preferences
                    saved_prefs = self.cache_manager.user_context_cache[cache_key]
                    self.cache_manager.cache_stats['hits']['user_context'] += 1
                    self.cache_manager.cache_stats['total_time_saved_ms'] += 70  # Avg DB lookup time
                    print(f"⚡ CACHE HIT: User preferences for {user_id} - Saved ~70ms")
                else:
                    # Cache miss - fetch from ChromaDB
                    print(f"🔍 Looking up preferences for user_id: {user_id}")
                    saved_prefs = self.chroma_manager.get_user_preferences_by_user_id(user_id)

                    # Store in cache if found
                    if saved_prefs:
                        self.cache_manager.user_context_cache[cache_key] = saved_prefs
                        print(f"✅ Found user preferences (cached for reuse): {saved_prefs}")

                    self.cache_manager.cache_stats['misses']['user_context'] += 1

                if saved_prefs:
                    user_preferences = saved_prefs
                else:
                    print(f"⚠️ No saved preferences found for user_id: {user_id}")
                    # Check if this is an event-finding request
                    event_keywords = ["event", "find", "recommend", "suggest", "show", "search", "looking for", "want"]
                    if any(keyword in query.lower() for keyword in event_keywords):
                        print(f"🚨 Event request detected without preferences - requesting preferences first")
                        return {
                            "success": False,
                            "recommendations": [],
                            "total_found": 0,
                            "message": "I'd love to help you find events! To give you personalized recommendations, please tell me what activities and interests you enjoy.",
                            "needs_preferences": True
                        }
            
            # Override with provided preferences
            if override_preferences:
                user_preferences.update(override_preferences)
            
            # IMPORTANT: Set location filter based on explicit request or current city
            if user_current_city:
                user_preferences['current_city'] = user_current_city

            # ============================================================================
            # NEW SMART QUERY ANALYSIS SYSTEM
            # ============================================================================
            # Use LLM-based query analyzer to understand user intent
            print(f"🔍 Analyzing query: '{query}' with saved prefs: {user_preferences}")
            query_analysis = self.analyze_user_query(query, user_preferences)
            self._record_checkpoint("Query analysis completed")

            # Extract analysis results
            query_activities = query_analysis.get('activities', [])
            query_location = query_analysis.get('location', '')
            is_specific_request = query_analysis.get('is_specific_request', False)
            should_override_saved = query_analysis.get('should_override_saved', False)

            print(f"📊 Query Analysis Results:")
            print(f"   - Activities: {query_activities}")
            print(f"   - Location: {query_location}")
            print(f"   - Is Specific: {is_specific_request}")
            print(f"   - Override Saved: {should_override_saved}")

            # ============================================================================
            # SEMANTIC ACTIVITY EXPANSION - Expand category terms to specific activities
            # ============================================================================
            # If LLM didn't extract activities, check raw query for category keywords
            if not query_activities:
                query_lower = query.lower()
                query_words = set(query_lower.split())

                # Find the BEST matching category (most overlapping words)
                best_match = None
                best_overlap_count = 0

                for category in self.activity_categories.keys():
                    # Check for partial word matches (e.g., "outdoor" matches "outdoor_sports")
                    category_words = set(category.replace('_', ' ').split())
                    overlap = category_words & query_words  # Set intersection

                    if overlap and len(overlap) > best_overlap_count:
                        best_match = category
                        best_overlap_count = len(overlap)

                    # Check for exact category match (substring)
                    category_normalized = category.replace('_', ' ')
                    if (category_normalized in query_lower or category in query_lower):
                        # Exact match gets highest priority (treat as full word count)
                        if len(category_words) > best_overlap_count:
                            best_match = category
                            best_overlap_count = len(category_words)

                # Use the best matching category
                if best_match:
                    query_activities = [best_match]
                    print(f"🔍 Detected category keyword in query: '{best_match}' ({best_overlap_count} word overlap)")
                    is_specific_request = True
                    should_override_saved = True

            # Expand activities (specific names or categories) to full lists
            # Track if this is a category-based search for smarter query construction
            is_category_search = False
            original_category = None

            if query_activities:
                expanded_activities = []
                for activity in query_activities:
                    # Check if this is a category name
                    if activity in self.activity_categories:
                        is_category_search = True
                        original_category = activity

                    similar = self.find_similar_activities(activity, max_results=10)
                    if similar:
                        expanded_activities.extend(similar)
                        if len(similar) > 1:
                            print(f"🔄 Expanded '{activity}' → {similar}")
                    else:
                        expanded_activities.append(activity)

                # Remove duplicates while preserving order
                seen = set()
                unique_activities = []
                for act in expanded_activities:
                    if act not in seen:
                        seen.add(act)
                        unique_activities.append(act)

                query_activities = unique_activities
                print(f"✨ Final expanded activities: {query_activities}")

                if is_category_search:
                    print(f"🏷️ Category search detected - will use '{original_category}' for semantic matching")

            # Get saved activities from preferences
            saved_activities = []
            if user_preferences and 'metadata' in user_preferences:
                activities_summary = user_preferences.get('metadata', {}).get('activities_summary', '')
                if activities_summary:
                    # Extract activities from summary (format: Club|ACTIVITY|City...)
                    activity_matches = re.findall(r'\|([A-Z_]+)\|', activities_summary)
                    saved_activities = [act.lower() for act in activity_matches]
                    print(f"💾 Saved activities from metadata: {saved_activities}")
            elif user_preferences:
                saved_activities = user_preferences.get('activities', [])
                print(f"💾 Saved activities: {saved_activities}")

            # ============================================================================
            # SMART PRIORITY SYSTEM FOR ACTIVITIES
            # ============================================================================
            final_activities = []
            search_query = query

            if should_override_saved and query_activities:
                # User explicitly requested different activities - OVERRIDE saved preferences
                final_activities = query_activities
                search_query = ' '.join(query_activities)
                print(f"🎯 OVERRIDE: Using query activities: {final_activities}")
            elif is_specific_request and query_activities:
                # User made a specific activity request - use it
                final_activities = query_activities
                search_query = ' '.join(query_activities)
                print(f"🎯 SPECIFIC: Using query activities: {final_activities}")
            elif saved_activities:
                # Generic request or no specific activities - use saved preferences
                final_activities = saved_activities
                search_query = ' '.join(saved_activities)
                print(f"💾 SAVED: Using saved activities: {final_activities}")
            else:
                # No saved preferences and no query activities - use full query
                search_query = query
                print(f"🔍 FALLBACK: Using full query: {search_query}")

            # Determine final search location
            final_location = query_location if query_location else (user_current_city if user_current_city else '')
            if final_location:
                search_query += f" {final_location}"
                print(f"📍 Adding location to search: {final_location}")

            # Build activity list for filtering/scoring
            activities = final_activities if final_activities else saved_activities
            explicit_activity_request = is_specific_request
            explicit_city_request = bool(query_location)
            
            # Activity types from database - EXACT MATCH to activity_type column
            # These are the actual values stored in the database
            all_activities_db = [
                'ART', 'BADMINTON', 'BASKETBALL', 'BOARDGAMING', 'BOOK_CLUB', 'BOWLING', 'BOX_CRICKET',
                'CHESS', 'COMMUNITY_SPACE', 'CONTENT_CREATION', 'CYCLING', 'DANCE', 'DEFAULT', 'DRAMA',
                'FILMS', 'FOOD', 'FOOTBALL', 'HARRY_POTTER', 'HIKING', 'HNI', 'INNER_JOURNEY',
                'JOURNALING', 'MINDFULNESS', 'MULTI_ACTIVITY_CLUB', 'MUSIC', 'OTHERS', 'PHOTOGRAPHY',
                'PICKLEBALL', 'POETRY', 'POP_CULTURE', 'QUIZ', 'RUNNING', 'SOCIAL_DEDUCTIONS',
                'TRAVEL', 'VIDEO_GAMES', 'VOLLEYBALL', 'WRITING', 'YOGA'
            ]

            # User-friendly activity mappings (user query → database activity_type)
            activity_name_mapping = {
                'art': 'ART',
                'badminton': 'BADMINTON',
                'basketball': 'BASKETBALL',
                'board gaming': 'BOARDGAMING',
                'boardgaming': 'BOARDGAMING',
                'board games': 'BOARDGAMING',
                'reading': 'BOOK_CLUB',
                'book club': 'BOOK_CLUB',
                'books': 'BOOK_CLUB',
                'bowling': 'BOWLING',
                'box cricket': 'BOX_CRICKET',
                'cricket': 'BOX_CRICKET',  # Map general cricket to box cricket
                'chess': 'CHESS',
                'community': 'COMMUNITY_SPACE',
                'community space': 'COMMUNITY_SPACE',
                'content creation': 'CONTENT_CREATION',
                'creator': 'CONTENT_CREATION',
                'cycling': 'CYCLING',
                'bike': 'CYCLING',
                'biking': 'CYCLING',
                'dance': 'DANCE',
                'drama': 'DRAMA',
                'theater': 'DRAMA',
                'theatre': 'DRAMA',
                'acting': 'DRAMA',
                'films': 'FILMS',
                'movies': 'FILMS',
                'cinema': 'FILMS',
                'food': 'FOOD',
                'cooking': 'FOOD',
                'culinary': 'FOOD',
                'football': 'FOOTBALL',
                'soccer': 'FOOTBALL',
                'harry potter': 'HARRY_POTTER',
                'hp': 'HARRY_POTTER',
                'potterhead': 'HARRY_POTTER',
                'hiking': 'HIKING',
                'trekking': 'HIKING',
                'trek': 'HIKING',
                'journaling': 'JOURNALING',
                'journal': 'JOURNALING',
                'mindfulness': 'MINDFULNESS',
                'meditation': 'MINDFULNESS',
                'wellness': 'MINDFULNESS',
                'music': 'MUSIC',
                'photography': 'PHOTOGRAPHY',
                'photo': 'PHOTOGRAPHY',
                'pickleball': 'PICKLEBALL',
                'pickle ball': 'PICKLEBALL',
                'poetry': 'POETRY',
                'poems': 'POETRY',
                'pop culture': 'POP_CULTURE',
                'popculture': 'POP_CULTURE',
                'quiz': 'QUIZ',
                'trivia': 'QUIZ',
                'running': 'RUNNING',
                'jogging': 'RUNNING',
                'social deduction': 'SOCIAL_DEDUCTIONS',
                'social deductions': 'SOCIAL_DEDUCTIONS',
                'mafia': 'SOCIAL_DEDUCTIONS',
                'werewolf': 'SOCIAL_DEDUCTIONS',
                'travel': 'TRAVEL',
                'video games': 'VIDEO_GAMES',
                'gaming': 'VIDEO_GAMES',
                'games': 'VIDEO_GAMES',
                'volleyball': 'VOLLEYBALL',
                'writing': 'WRITING',
                'creative writing': 'WRITING',
                'yoga': 'YOGA',
                'fitness': 'YOGA',  # Could map to multiple
                'tech': 'COMMUNITY_SPACE',  # Map tech to community space or others
                'technology': 'COMMUNITY_SPACE',
                'coding': 'COMMUNITY_SPACE',
                'startup': 'COMMUNITY_SPACE',
                'business': 'COMMUNITY_SPACE',
                'networking': 'COMMUNITY_SPACE'
            }

            # Normalize activities for database queries
            def normalize_activity(activity: str) -> str:
                """Convert user-friendly activity name to database activity_type"""
                activity_lower = activity.lower().strip()
                return activity_name_mapping.get(activity_lower, activity.upper())
            
            city_keywords = ['mumbai', 'delhi', 'bangalore', 'bengaluru', 'pune', 'chennai', 'kolkata',
                           'noida', 'gurgaon', 'gurugram', 'hyderabad', 'ahmedabad', 'faridabad', 'ghaziabad',
                           'new delhi', 'bombay', 'madras', 'calcutta']
            
            # Activity mapping for better matching - maps user request to database activities
            activity_mappings = {
                'football': ['football', 'soccer', 'sports', 'outdoor_sports'],
                'soccer': ['football', 'soccer', 'sports'],
                'cricket': ['cricket', 'sports', 'outdoor_sports'],
                'badminton': ['badminton', 'sports', 'indoor_sports'],
                'tennis': ['tennis', 'sports', 'outdoor_sports'],
                'basketball': ['basketball', 'sports', 'outdoor_sports'],
                'volleyball': ['volleyball', 'sports', 'outdoor_sports'],
                'swimming': ['swimming', 'sports', 'fitness'],
                'gym': ['gym', 'fitness', 'workout'],
                'yoga': ['yoga', 'fitness', 'wellness'],
                'dance': ['dance', 'arts', 'fitness'],
                'music': ['music', 'arts', 'entertainment'],
                'art': ['art', 'arts', 'creative'],
                'tech': ['tech', 'technology', 'coding', 'startup'],
                'coding': ['coding', 'tech', 'technology', 'programming'],
                'quiz': ['quiz', 'trivia', 'knowledge'],
                'drama': ['drama', 'theater', 'theatre', 'acting'],
                'theater': ['theater', 'theatre', 'drama', 'acting'],
                'sports': ['sports', 'football', 'cricket', 'badminton', 'tennis', 'basketball'],
                'boardgaming': ['boardgaming', 'board_games', 'board games', 'games', 'tabletop'],
                'social_deductions': ['social_deductions', 'social deductions', 'deduction', 'mafia', 'werewolf'],
                'book_club': ['book_club', 'book club', 'reading', 'books', 'literature'],
                'box_cricket': ['box_cricket', 'box cricket', 'cricket', 'sports'],
                'films': ['films', 'movies', 'cinema', 'movie', 'film'],
                'poetry': ['poetry', 'poems', 'poem', 'writing', 'literature'],
                'writing': ['writing', 'creative writing', 'storytelling', 'literature'],
                'harry_potter': ['harry_potter', 'harry potter', 'hp', 'hogwarts', 'potterhead'],
                'pop_culture': ['pop_culture', 'pop culture', 'popculture', 'entertainment'],
                'community_space': ['community_space', 'community space', 'community', 'social'],
                'content_creation': ['content_creation', 'content creation', 'creator', 'creative'],
                'bowling': ['bowling', 'sports', 'entertainment'],
                'mindfulness': ['mindfulness', 'meditation', 'wellness', 'mental health'],
                'others': ['others', 'other', 'miscellaneous', 'misc'],
                'pickleball': ['pickleball', 'pickle ball', 'sports'],
                'journaling': ['journaling', 'journal', 'writing', 'reflection'],
                'trekking': ['trekking', 'hiking', 'outdoor', 'adventure'],
                'cycling': ['cycling', 'biking', 'bike', 'sports'],
                'photography': ['photography', 'photo', 'creative', 'art'],
                'hiking': ['hiking', 'trekking', 'outdoor', 'adventure'],
                'fitness': ['fitness', 'workout', 'gym', 'exercise'],
                'improv': ['improv', 'improvisation', 'comedy', 'theater'],
                'startup': ['startup', 'business', 'entrepreneur', 'tech'],
                'business': ['business', 'networking', 'startup', 'entrepreneur'],
                'networking': ['networking', 'business', 'professional'],
                'cooking': ['cooking', 'food', 'culinary', 'chef'],
                'food': ['food', 'cooking', 'culinary', 'dining']
            }
            
            # STEP 1: Extract explicitly mentioned activities and cities from current query
            query_lower = query.lower()
            explicit_activities = []
            explicit_city = None
            
            # Find explicitly mentioned activities
            for activity in all_activities_db:
                if activity in query_lower:
                    explicit_activities.append(activity)
                    explicit_activity_request = True
                    print(f"🎯 Found explicit activity in query: {activity}")

            # Add activities from category expansion to explicit_activities
            if is_specific_request and activities:
                # Normalize expanded activities to database format and add to explicit_activities
                for activity in activities:
                    normalized = activity_name_mapping.get(activity.lower(), activity.upper())
                    if normalized not in explicit_activities:
                        explicit_activities.append(normalized)
                print(f"🎯 Added expanded activities to explicit filter: {explicit_activities}")

            # Find explicitly mentioned city
            for city in city_keywords:
                if city in query_lower:
                    explicit_city = city
                    explicit_city_request = True
                    print(f"🏙️ Found explicit city in query: {city}")
                    break
            
            # STEP 2: Get saved preferences if available (but don't override explicit requests)
            if user_preferences and not should_override_saved:
                print(f"🔍 Processing user preferences: {user_preferences}")
                # Extract activities from user preferences (from metadata if available)
                if 'metadata' in user_preferences:
                    activities_summary = user_preferences.get('metadata', {}).get('activities_summary', '')
                    print(f"📝 Activities summary from metadata: {activities_summary}")
                    # Extract specific activities from the summary
                    if activities_summary:
                        # Parse the activities_summary format: "Club|ACTIVITY|City|Area|Count"
                        activity_matches = re.findall(r'\|([A-Z_]+)\|', activities_summary)
                        for activity in activity_matches:
                            if activity.lower() not in activities:
                                activities.append(activity.lower())

                        # Fallback: check for common activity keywords in summary
                        for keyword in all_activities_db:
                            if keyword.lower() in activities_summary.lower() and keyword.lower() not in activities:
                                activities.append(keyword.lower())
                else:
                    saved_prefs_activities = user_preferences.get('activities', [])
                    for activity in saved_prefs_activities:
                        if activity not in activities:
                            activities.append(activity)

                print(f"💾 Saved activities combined with query: {activities}")
            elif user_preferences and should_override_saved:
                print(f"🎯 Skipping saved preferences (user made explicit request)")
            
            # STEP 3: Determine search city priority
            if explicit_city:
                search_city = explicit_city
                print(f"🏙️ Using explicitly requested city: {explicit_city}")
            elif user_current_city:
                search_city = user_current_city
                print(f"📍 Using current city: {user_current_city}")
            else:
                search_city = user_preferences.get('location', '') if user_preferences else ''
                print(f"💾 Using saved location preference: {search_city}")
            
            # STEP 4: Build search query based on priority
            is_generic_request = any(phrase in query_lower for phrase in [
                'find events', 'show me events', 'events for me', 'recommend events',
                'suggest events', 'looking for events', 'what events'
            ]) or query_lower.strip() == 'events'
            
            if explicit_activities:
                # User explicitly requested specific activities - highest priority
                # For category searches, use category name (semantic) instead of expanded list
                if is_category_search and original_category:
                    # Use natural category name for better semantic matching
                    category_natural = original_category.replace('_', ' ')
                    search_query = category_natural
                    if search_city:
                        search_query += f" {search_city}"
                    print(f"🎯 Using CATEGORY semantic search: '{search_query}' (will filter by: {explicit_activities})")
                else:
                    # Use lowercase natural language for vector search (not DB format)
                    search_query = ' '.join(activities) if activities else ' '.join([act.lower() for act in explicit_activities])
                    if search_city:
                        search_query += f" {search_city}"
                    print(f"🎯 Using EXPLICIT activity request: {search_query}")
            elif explicit_city and not explicit_activities:
                # User asked for different city but no specific activity
                if activities and is_generic_request:
                    # Use saved activity preferences with requested city
                    search_query = ' '.join(activities) + f" {explicit_city}"
                    print(f"🏙️ Using saved activities with explicit city: {search_query}")
                else:
                    # Generic events in requested city
                    search_query = f"events {explicit_city}"
                    print(f"🏙️ Generic events in explicit city: {search_query}")
            elif is_generic_request and activities:
                # Generic request like "show me events" - use saved preferences
                search_query = ' '.join(activities)
                if search_city:
                    search_query += f" {search_city}"
                print(f"💾 Using SAVED preference-based search query: {search_query}")
            else:
                # Default: use query as-is with appropriate city
                search_query = query
                if search_city and search_city not in query_lower:
                    search_query += f" {search_city}"
                print(f"🔎 Using original query with city: {search_query}")
            
            # Set location preference based on search_city determined above
            if search_city:
                user_preferences['location'] = search_city
                print(f"📍 Set location preference to: {search_city}")

            # ============================================================================
            # VECTOR SEARCH CACHING - Avoid expensive ChromaDB vector search for repeat queries
            # ============================================================================
            relevant_events = []
            if search_query:
                # Generate cache key for vector search (based on query + location + limit)
                vsearch_cache_key = f"vsearch:{self.cache_manager.generate_cache_key(search_query, final_location, limit)}"

                if vsearch_cache_key in self.cache_manager.vector_search_cache:
                    # Cache hit - use cached search results
                    relevant_events = self.cache_manager.vector_search_cache[vsearch_cache_key]
                    self.cache_manager.cache_stats['hits']['vector_search'] += 1
                    self.cache_manager.cache_stats['total_time_saved_ms'] += 150  # Avg vector search time
                    print(f"⚡ CACHE HIT: Vector search ({len(relevant_events)} events) - Saved ~150ms")
                else:
                    # Cache miss - perform actual vector search WITH date filtering
                    # Filter by start_timestamp in ChromaDB to exclude past events BEFORE vector search
                    from datetime import datetime

                    # Get current timestamp for filtering
                    current_timestamp = datetime.now().timestamp()

                    # ChromaDB filter: start_timestamp >= current_time (numeric comparison)
                    date_filter = {
                        "start_timestamp": {"$gte": current_timestamp}  # Only upcoming events
                    }

                    print(f"🔍 Performing vector search for: '{search_query}' (upcoming events only)")
                    print(f"📅 Date filter: start_timestamp >= {current_timestamp}")
                    relevant_events = self.chroma_manager.search_events(
                        search_query,
                        n_results=limit * 3,  # 3x multiplier - category semantic search ensures quality
                        filters=date_filter
                    )

                    # Store in cache
                    self.cache_manager.vector_search_cache[vsearch_cache_key] = relevant_events
                    self.cache_manager.cache_stats['misses']['vector_search'] += 1
                    print(f"✅ Found {len(relevant_events)} events (cached for reuse)")

            self._record_checkpoint("Vector search completed")

            # Filter out past events by parsing start_time field
            # (Not using ChromaDB filter to support events without start_timestamp field)
            from datetime import datetime
            current_time = datetime.now()
            initial_count = len(relevant_events)
            future_events = []

            for event in relevant_events:
                try:
                    start_time_str = event.get('start_time', '')
                    if start_time_str:
                        # Parse datetime (handle both "2025-09-10 15:00" and "2025-09-10 15:00 IST")
                        try:
                            event_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M IST")
                        except ValueError:
                            try:
                                event_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
                            except ValueError:
                                # If can't parse, include it
                                future_events.append(event)
                                continue

                        if event_time > current_time:
                            future_events.append(event)
                    else:
                        # No start time, include it
                        future_events.append(event)
                except Exception:
                    # Parsing failed, include it
                    future_events.append(event)

            relevant_events = future_events
            past_filtered = initial_count - len(relevant_events)
            if past_filtered > 0:
                print(f"🗓️ Filtered out {past_filtered} past events")
            print(f"📅 {len(relevant_events)} upcoming events ready for ranking")
            
            # ⭐ EXPLICIT ACTIVITY VALIDATION - When user explicitly requests activities, validate results match
            if explicit_activity_request and explicit_activities:
                print(f"🔍 Validating events match explicit request: {explicit_activities}")
                matched_events = []
                
                for event in relevant_events:
                    event_activity = event.get('activity', '').lower()
                    event_matches = False
                    
                    # Check if event matches any requested activity
                    for requested_activity in explicit_activities:
                        # Direct match
                        if requested_activity.lower() in event_activity or event_activity in requested_activity.lower():
                            event_matches = True
                            print(f"✅ Direct match: {event.get('name', '')} - {event_activity} matches {requested_activity}")
                            break
                        
                        # Check through activity mappings
                        mapped_activities = activity_mappings.get(requested_activity.lower(), [])
                        for mapped_activity in mapped_activities:
                            if mapped_activity.lower() in event_activity or event_activity in mapped_activity.lower():
                                event_matches = True
                                print(f"✅ Mapped match: {event.get('name', '')} - {event_activity} matches {requested_activity} via {mapped_activity}")
                                break
                        
                        if event_matches:
                            break
                    
                    if event_matches:
                        matched_events.append(event)
                    else:
                        print(f"❌ No match: {event.get('name', '')} - {event_activity} doesn't match {explicit_activities}")
                
                relevant_events = matched_events
                print(f"🎯 After explicit activity validation: {len(relevant_events)} events remain")
                
                # If no events match the explicit request, return appropriate message
                if len(relevant_events) == 0:
                    requested_activities_str = ', '.join(explicit_activities)
                    # Use the guidance message generator for better response
                    guidance_message = self._generate_no_events_guidance_message_for_activity(
                        search_city, requested_activities_str, user_id
                    )
                    return {
                        "success": True,
                        "recommendations": [],
                        "total_found": 0,
                        "message": guidance_message,  # Use guidance directly as message
                        "user_preferences_used": user_preferences
                    }
            
            # ⭐ SMART ACTIVITY PREFERENCE FILTERING - Only apply for generic requests, not explicit ones
            if user_preferences and 'metadata' in user_preferences and not explicit_activity_request and not explicit_city_request:
                activities_summary = user_preferences.get('metadata', {}).get('activities_summary', '')
                user_preferred_activities = []
                if activities_summary:
                    # Extract preferred activities from summary format: "Club|ACTIVITY|City|Area|Count"
                    activity_matches = re.findall(r'\|([A-Z_]+)\|', activities_summary)
                    for activity in activity_matches:
                        user_preferred_activities.append(activity.lower())
                    
                    # Fallback: Extract preferred activities from summary - includes all Misfits activities
                    activity_keywords = ['cricket', 'football', 'badminton', 'tennis', 'swimming', 'gym', 'yoga', 
                                       'dance', 'music', 'art', 'photography', 'hiking', 'trekking', 'cycling',
                                       'tech', 'coding', 'startup', 'business', 'networking', 'food', 'cooking', 
                                       'pickleball', 'journaling', 'quiz', 'drama', 'theater', 'improv',
                                       'boardgaming', 'social_deductions', 'book_club', 'box_cricket', 'films', 
                                       'poetry', 'writing', 'harry_potter', 'pop_culture', 'community_space', 
                                       'content_creation', 'bowling', 'mindfulness', 'others', 'sports', 'fitness']
                    for keyword in activity_keywords:
                        if keyword.lower() in activities_summary.lower() and keyword.lower() not in user_preferred_activities:
                            user_preferred_activities.append(keyword.lower())
                
                if user_preferred_activities:
                    print(f"🔒 Applying strict activity filtering for: {user_preferred_activities}")
                    activity_filtered_events = []
                    for event in relevant_events:
                        event_activity = event.get('activity', '').lower()
                        # Check if event activity matches any of user's preferred activities
                        if any(pref_activity in event_activity or event_activity in pref_activity for pref_activity in user_preferred_activities):
                            activity_filtered_events.append(event)
                        else:
                            print(f"🚫 Filtered out non-matching activity: {event.get('activity', '')} for {event.get('name', '')}")
                    
                    relevant_events = activity_filtered_events
                    print(f"🎯 After activity preference filtering: {len(relevant_events)} events remain")
            
            # Filter by location (either requested city or current city)
            filter_city = user_preferences.get('location', user_current_city)
            if filter_city:
                # Strict city filtering - only show events in the target city
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
                
                # Get variations for the filter city (either requested or current)
                filter_city_variations = city_variations.get(filter_city, [filter_city])
                
                for event in relevant_events:
                    event_city = event.get('city_name', '').lower().strip()
                    
                    # Check if event city matches any variation of filter city
                    city_match = False
                    for variation in filter_city_variations:
                        if variation in event_city or event_city in variation:
                            city_match = True
                            break
                    
                    # Also check exact match
                    if not city_match and (filter_city in event_city or event_city in filter_city):
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
            
            # Handle empty state - if no relevant events found
            if not relevant_events:
                # Use conversational message based on context
                conversational_message = self._generate_conversational_no_events_message(
                    query, filter_city, explicit_activities if explicit_activity_request else [], user_id
                )
                return {
                    "success": False,  # Changed to False to indicate no events found
                    "recommendations": [],
                    "total_found": 0,
                    "message": conversational_message
                }
            
            # For explicit requests, skip scoring - user knows exactly what they want
            if explicit_activity_request and explicit_activities:
                print(f"🚀 Explicit request detected - bypassing scoring for: {explicit_activities}")

                # DIVERSIFICATION LOGIC: For category searches, diversify results by activity
                events_to_format = []
                if is_category_search and len(explicit_activities) > 1:
                    # Category search (e.g., "outdoor sports") - diversify results
                    print(f"🎨 Category search detected - diversifying results across {len(explicit_activities)} activities")

                    # Group events by activity
                    activity_groups = {}
                    for event in relevant_events:
                        activity = event.get('activity', '').upper()
                        if activity not in activity_groups:
                            activity_groups[activity] = []
                        activity_groups[activity].append(event)

                    # Show activity distribution
                    activity_counts = {act: len(events) for act, events in activity_groups.items()}
                    print(f"📊 Activity distribution: {activity_counts}")

                    # Adjust max_per_activity based on number of activity types
                    num_activity_types = len(activity_groups)
                    if num_activity_types == 1:
                        # Only one activity type - use all available up to limit
                        max_per_activity = limit
                        print(f"⚠️ Only 1 activity type available - will return up to {limit} events")
                    elif num_activity_types == 2:
                        # Two activities - allow more per activity
                        max_per_activity = max(3, limit // 2 + 1)
                    else:
                        # Multiple activities - limit to 3 per activity for diversity
                        max_per_activity = 3

                    print(f"🎯 Max events per activity: {max_per_activity}")
                    events_per_activity = {}  # Track how many we've taken from each

                    # First pass: take 1 from each activity (ensure variety)
                    for activity in sorted(activity_groups.keys()):
                        if len(events_to_format) >= limit:
                            break
                        if activity_groups[activity]:
                            events_to_format.append(activity_groups[activity].pop(0))
                            events_per_activity[activity] = 1
                            print(f"✅ Added 1 {activity} event (diversity pass)")

                    # Second pass: fill remaining slots round-robin (max 3 per activity)
                    while len(events_to_format) < limit:
                        added_any = False
                        for activity in sorted(activity_groups.keys()):
                            if len(events_to_format) >= limit:
                                break
                            # Check if we haven't exceeded max per activity
                            if events_per_activity.get(activity, 0) < max_per_activity and activity_groups[activity]:
                                events_to_format.append(activity_groups[activity].pop(0))
                                events_per_activity[activity] = events_per_activity.get(activity, 0) + 1
                                added_any = True
                                print(f"✅ Added 1 more {activity} event (count: {events_per_activity[activity]})")

                        # If no more events can be added, break
                        if not added_any:
                            break

                    # Show final distribution
                    final_distribution = {}
                    for event in events_to_format:
                        activity = event.get('activity', '').upper()
                        final_distribution[activity] = final_distribution.get(activity, 0) + 1
                    print(f"🎯 Final event distribution: {final_distribution}")

                else:
                    # Specific activity search (e.g., "badminton events") - return all from that activity
                    print(f"🎯 Specific activity search - returning top {limit} events")
                    events_to_format = relevant_events[:limit]

                # Format the selected events
                formatted_events = []
                for event in events_to_format:
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

                    # Get event name with proper fallbacks
                    event_name = event.get('name', '').strip()
                    if not event_name:
                        event_name = event.get('event_name', '').strip()
                    if not event_name:
                        event_name = f"{event.get('activity', 'Event')} at {event.get('location_name', 'Venue')}"

                    event_data = {
                        "event_id": event.get('event_id', ''),
                        "name": event_name,
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
                        "price": self._safe_float(event.get('ticket_price', 0)),
                        "available_spots": self._safe_int(event.get('available_spots', 0)),
                        "registration_url": registration_url,
                        # New fields from API
                        "activity_icon_url": event.get('activity_icon_url', ''),
                        "club_icon_url": event.get('club_icon_url', ''),
                        "event_cover_image_url": event.get('event_cover_image_url', ''),
                        "event_uuid": event.get('event_uuid', ''),
                        "participants_count": self._safe_int(event.get('participants_count', 0))
                    }
                    formatted_events.append(event_data)

                requested_activities_str = ', '.join(explicit_activities)
                # More conversational success message
                if len(formatted_events) == 1:
                    success_message = f"Great! Found a perfect {requested_activities_str} event for you in {filter_city}!"
                else:
                    success_message = f"Awesome! Found {len(formatted_events)} {requested_activities_str} events in {filter_city}!"

                return {
                    "success": True,
                    "recommendations": formatted_events,
                    "total_found": len(relevant_events),
                    "message": success_message
                }
            
            # Score and format events (for non-explicit requests)
            # Pass query_activities to scoring function for boost
            scored_events = []
            for event in relevant_events:
                score = self._calculate_match_score_enhanced(event, user_preferences, filter_city, query_activities=query_activities)
                
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
                    # Get event name with proper fallbacks
                    event_name = event.get('name', '').strip()
                    if not event_name:
                        event_name = event.get('event_name', '').strip()
                    if not event_name:
                        event_name = f"{event.get('activity', 'Event')} at {event.get('location_name', 'Venue')}"
                    
                    event_data = {
                        "event_id": event.get('event_id', ''),
                        "name": event_name,
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
                        "price": self._safe_float(event.get('ticket_price', 0)),
                        "available_spots": self._safe_int(event.get('available_spots', 0)),
                        "registration_url": registration_url,
                        # New fields from API
                        "activity_icon_url": event.get('activity_icon_url', ''),
                        "club_icon_url": event.get('club_icon_url', ''),
                        "event_cover_image_url": event.get('event_cover_image_url', ''),
                        "event_uuid": event.get('event_uuid', ''),
                        "participants_count": self._safe_int(event.get('participants_count', 0)),
                        "_score": score  # Temporary for sorting
                    }
                    scored_events.append(event_data)
            
            # Sort by score (highest first)
            scored_events.sort(key=lambda x: x['_score'], reverse=True)
            
            # Remove internal score field from all events
            for event_data in scored_events:
                event_data.pop('_score', None)
            top_recommendations = scored_events[:limit]
            
            # More conversational success message for general recommendations
            if len(top_recommendations) == 0:
                # This case should be handled by the empty state above, but just in case
                conversational_message = self._generate_conversational_no_events_message(
                    query, filter_city, explicit_activities if explicit_activity_request else [], user_id
                )
                message = conversational_message
            elif len(top_recommendations) == 1:
                message = f"Perfect! Found a great event that matches your interests in {filter_city}!" if filter_city else "Perfect! Found a great event that matches your interests!"
            else:
                message = f"Excellent! Found {len(top_recommendations)} events that match your interests in {filter_city}!" if filter_city else f"Excellent! Found {len(top_recommendations)} events that match your interests!"

            self._record_checkpoint("Event scoring and ranking completed")

            # Prepare final response
            final_response = {
                "success": True,
                "recommendations": top_recommendations,
                "total_found": len(scored_events),
                "message": message
            }

            # ============================================================================
            # TIER 4: Store result in cache before returning
            # ============================================================================
            self.cache_manager.results_cache[results_cache_key] = final_response
            self._record_checkpoint("Result stored in cache")
            print(f"💾 Stored complete response in Tier 4 cache for future requests")

            # Get performance report for logging
            perf_report = self._get_performance_report()
            print(f"⏱️ Total request time: {perf_report.get('total_ms', 0)}ms")

            return final_response
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recommendations": [],
                "message": f"Error: {str(e)}"
            }

    def _calculate_match_score_enhanced(self, event: dict, preferences: dict, current_city: str = "", query_activities: list = None) -> float:
        """
        Enhanced scoring with dynamic city preference and query activity boost

        Args:
            event: Event data dict
            preferences: User preferences dict
            current_city: Current city for location filtering
            query_activities: Activities explicitly requested in current query (HIGHEST PRIORITY)

        Returns:
            Match score between 0.0 and 1.0
        """
        score = 0.0
        query_activities = query_activities or []

        # City match (highest priority - 30% weight)
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
                    score += 0.30
                    break

            # Also check direct match
            if score == 0 and (current_city in event_city or event_city in current_city):
                score += 0.30

        # ============================================================================
        # QUERY ACTIVITY MATCH (HIGHEST PRIORITY - 50% weight)
        # ============================================================================
        # If user explicitly requested activities in current query, prioritize those
        event_activity = event.get('activity', '').lower()
        query_match_found = False

        if query_activities:
            for query_activity in query_activities:
                query_act_lower = query_activity.lower()
                if query_act_lower in event_activity or event_activity in query_act_lower:
                    score += 0.50  # MAXIMUM boost for query-matched activities
                    query_match_found = True
                    print(f"🎯 QUERY MATCH BOOST: +0.50 for {event.get('name', '')} matching {query_activity}")
                    break

        # ============================================================================
        # SAVED PREFERENCE ACTIVITY MATCH (25% weight - only if no query match)
        # ============================================================================
        # Only use saved preferences if query didn't match (prevents double scoring)
        if not query_match_found:
            user_activities = []
            # Extract activities from metadata if available
            if 'metadata' in preferences:
                activities_summary = preferences.get('metadata', {}).get('activities_summary', '')
                if activities_summary:
                    # Extract activities from the summary format: "Club|ACTIVITY|City|Area|Count"
                    activity_matches = re.findall(r'\|([A-Z_]+)\|', activities_summary)
                    for activity in activity_matches:
                        user_activities.append(activity.lower())
            else:
                user_activities = preferences.get('activities', [])

            if user_activities:
                for activity in user_activities:
                    if activity.lower() in event_activity or event_activity in activity.lower():
                        score += 0.25  # Moderate weight for saved preference match
                        print(f"💾 SAVED PREF MATCH: +0.25 for {event.get('name', '')} matching {activity}")
                        break

        # Area match (10% weight)
        user_areas = [area.lower() for area in preferences.get('areas', [])]
        event_area = event.get('area_name', '').lower()
        if user_areas and event_area in user_areas:
            score += 0.10

        # Availability (5% weight)
        available_spots = event.get('available_spots', 0)
        # Ensure numeric type
        try:
            available_spots = int(available_spots) if isinstance(available_spots, str) else available_spots
        except (ValueError, TypeError):
            available_spots = 0
        if available_spots > 0:
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

    def _generate_no_city_events_message(self, city: str, query: str) -> list:
        """Message when no events found in requested city"""
        import random
        
        # Extract activity from query for personalized message
        activity = ""
        for activity_word in ['football', 'soccer', 'cricket', 'badminton', 'tennis', 'basketball', 
                             'volleyball', 'swimming', 'gym', 'yoga', 'quiz', 'drama', 'sports', 
                             'music', 'tech', 'dance', 'comedy', 'art', 'hiking', 'cycling', 
                             'running', 'fitness', 'pickleball', 'theater', 'improv', 'boardgaming',
                             'social_deductions', 'book_club', 'box_cricket', 'films', 'poetry', 
                             'writing', 'harry_potter', 'pop_culture', 'community_space', 
                             'content_creation', 'bowling', 'mindfulness', 'others']:
            if activity_word in query.lower():
                activity = activity_word
                break
        
        if activity:
            # Activity-specific empathetic message variations
            responses = [
                f"Oh, I checked everywhere for {activity} events in {city.title()} but came up empty today 😊 Want me to check what's happening nearby?",
                f"Hmm, {city.title()}'s {activity} scene is quiet right now 🤔 How about exploring something different today?",
                f"Nothing's happening in {city.title()} for {activity} today - but hey, sometimes the best discoveries come from trying something new! ✨",
                f"Looks like {activity} events in {city.title()} are taking a little break 🌙 Ready to discover a new favorite activity?",
                f"No {activity} vibes in {city.title()} right now, but I've got some other cool ideas! What do you think? 🎯"
            ]
            
            return random.choice(responses)
        else:
            # Generic city message without specific activity - more conversational
            responses = [
                f"Hmm, that's a tough one to find in {city.title()} right now 🤷‍♀️ What else sounds fun to you?",
                f"Nothing's popping up for that in {city.title()} today. Tell me more about what you're in the mood for?",
                f"Oh, I'm not seeing anything like that in {city.title()} right now. What would make your perfect day out?",
                f"Looks like {city.title()} doesn't have what you're looking for today. Want to try a different angle? 🎯",
                f"That's not showing up in {city.title()} right now 😊 What kind of vibe are you going for instead?"
            ]
            
            return random.choice(responses)

    def _generate_no_events_guidance_message_for_activity(self, city: str, activity: str, user_id: str = None):
        """Generate helpful message when specific activity has no events"""
        import random
        
        # Check if user has saved preferences
        has_preferences = False
        if user_id:
            try:
                user_prefs = self.chroma_manager.get_user_preferences_by_user_id(user_id)
                has_preferences = bool(user_prefs)
            except Exception:
                pass
        
        if not has_preferences:
            # User has no preferences - ask for them
            messages = [
                f"I couldn't find any {activity} events in {city.title()} right now. Could you tell me more about your preferred activities and areas? This will help me find better matches for you!",
                f"No {activity} events available in {city.title()} today. What other activities do you enjoy? Also, which areas of {city.title()} work best for you?",
                f"Looks like {activity} isn't happening in {city.title()} at the moment. Let me know your other interests and preferred locations so I can suggest great alternatives!"
            ]
        else:
            # User has preferences - suggest alternatives
            messages = [
                f"No {activity} events in {city.title()} right now, but I have other great options based on your interests! Would you like to see what's available?",
                f"I couldn't find {activity} in {city.title()} today. How about checking out events in nearby cities, or trying a different activity?",
                f"{activity.title()} isn't available in {city.title()} at the moment. Would you like me to show you events from your saved preferences instead?"
            ]
        
        return random.choice(messages)

    def _generate_conversational_no_events_message(self, query: str, city: str, activities: list, user_id: str = None):
        """Generate helpful conversational message when no events found based on context"""
        import random
        
        # Determine context
        has_activity = bool(activities)
        has_city = bool(city)
        query_lower = query.lower()
        
        if has_activity and has_city:
            # Both activity and city specified
            activity_str = ', '.join(activities)
            messages = [
                f"We couldn't find any {activity_str} events in {city.title()} right now. Would you be interested in trying a different activity or checking events in nearby cities?",
                f"No {activity_str} events available in {city.title()} at the moment. How about exploring other activities or different locations?",
                f"Looks like {activity_str} isn't happening in {city.title()} currently. Can I suggest similar activities or events in other cities?"
            ]
        elif has_city and not has_activity:
            # Only city specified
            messages = [
                f"We couldn't find any events in {city.title()} matching your criteria. What activities are you interested in? This will help me find better options.",
                f"No events found in {city.title()} right now. Would you like to check events in nearby cities or tell me what activities you enjoy?",
                f"Nothing available in {city.title()} at the moment. Should I look in different cities or would you like to specify what type of events you're looking for?"
            ]
        elif has_activity and not has_city:
            # Only activity specified
            activity_str = ', '.join(activities)
            messages = [
                f"We couldn't find any {activity_str} events in your area. Would you like to search in a different city or try other activities?",
                f"No {activity_str} events available right now. Can I suggest different activities or should I check other locations?",
                f"Looks like {activity_str} isn't available at the moment. Would you be interested in similar activities or events in other areas?"
            ]
        else:
            # Generic request
            messages = [
                "We couldn't find any events matching your request. Could you tell me what activities interest you and your preferred location?",
                "No events found for your criteria. What type of activities do you enjoy? Also, which city or area works best for you?",
                "I couldn't find suitable events right now. Help me understand your interests - what activities and locations would you prefer?"
            ]
        
        # Add time-based suggestions if weekend/weekday mentioned
        if any(time_word in query_lower for time_word in ['weekend', 'saturday', 'sunday', 'weekday', 'today', 'tomorrow']):
            messages = [m.rstrip('?') + " Or would you like to see events for different dates?" for m in messages]
        
        return random.choice(messages)
    
    def _generate_no_preferences_message(self, city: str) -> list:
        """Message when user has no saved preferences"""
        import random
        
        openings = [
            f"Hey! I'd love to help you discover amazing events in {city.title()}! 🎉",
            f"Hi there! Ready to explore what {city.title()} has to offer? 🌟",
            f"Welcome! Let's find you some fantastic events in {city.title()}! 🚀"
        ]
        
        endings = [
            f"Just tell me what you're into and I'll find some amazing events for you!",
            f"Share your interests and I'll discover the perfect activities! ✨",
            f"Let me know what excites you and I'll match you with great events! 🎯"
        ]
        
        opening = random.choice(openings)
        ending = random.choice(endings)
        
        return [
            opening,
            "To find the perfect matches for you, tell me:",
            "What do you enjoy? (sports, music, comedy, tech talks, art...)",
            "When are you free? (weekends, evenings, mornings...)",
            "What's your vibe? (chill hangouts, competitive games, learning something new...)",
            ending
        ]
    
    def _generate_vague_search_message(self, city: str) -> list:
        """Message when search query is too vague"""
        import random
        
        responses = [
            "Tell me more! What kind of vibe are you going for? 😊",
            "I need a bit more to work with - what sounds fun to you right now? 🎯",
            "What would make your perfect day out? Give me some hints! ✨",
            "Hmm, help me out here - what's your mood today? Adventurous? Chill? Creative? 🤔",
            "I want to find you something amazing! What are you feeling like doing? 🎆",
            "You know what? Let's get specific! What kind of experience are you after? 🎭"
        ]
        
        return random.choice(responses)
    
    def _generate_no_time_match_message(self, city: str, time_constraint: str) -> list:
        """Message when no events match time preferences"""
        import random
        
        responses = [
            f"I get it, {time_constraint} works best for you. Nothing's scheduled then, but I've got some other great times that might work! 😊",
            f"Your {time_constraint} is precious - while nothing's happening then, how about these other options? ⏰",
            f"Ah, {time_constraint} person! I respect that. Let me show you what's available at other times 🌅",
            f"I totally understand the {time_constraint} preference! Here are some flexible alternatives that might fit 🏃‍♂️",
            f"No {time_constraint} events right now, but don't worry - I found some other awesome timing options! ✨"
        ]
        
        return random.choice(responses)
    
    def _generate_budget_constraint_message(self, city: str, budget: str) -> list:
        """Message when no events within budget"""
        import random
        
        responses = [
            f"I totally understand wanting to keep it affordable! Nothing under {budget} right now, but I've got some wallet-friendly alternatives 💰",
            f"Being smart with money - I respect that! While {budget} events aren't available, here are some great budget options 😊",
            f"Hey, I get it - budget matters! No {budget} events today, but let me show you what's doable 🤝",
            f"Love that you're being budget-conscious! Nothing in the {budget} range, but I found some affordable gems 💎",
            f"Smart budgeting! While {budget} events aren't happening, I've got some reasonably-priced options that might work 🎯"
        ]
        
        return random.choice(responses)
    
    def _generate_generic_empty_message(self, city: str) -> list:
        """Generic friendly message for empty results"""
        import random
        
        responses = [
            "Hmm, that's a tough one to find right now 🤷‍♀️ Let me think of something else you might enjoy...",
            "You know what? Let's try a different angle - what kind of experience are you after? 🎭",
            "That's not showing up for me today 😊 What would make your perfect day out?",
            "Let me think outside the box here! What sounds fun to you right now? ✨",
            "Hmm, nothing's coming up for that specific thing. Tell me more about what you're feeling like doing? 🎯",
            "That's a tricky one! How about we explore what else might spark your interest? 🚀"
        ]
        
        return random.choice(responses)

    def _generate_no_events_guidance_message(self, city: str, query: str, user_id: str = None):
        """Generate helpful guidance when no events are found"""
        
        # Check if user has preferences saved
        if user_id:
            try:
                user_prefs = self.chroma_manager.get_user_preferences_by_user_id(user_id)
                if not user_prefs:
                    message_list = self._generate_no_preferences_message(city)
                    return message_list  # Return list directly
            except Exception:
                pass
        
        # Determine the type of null state based on query and context
        query_lower = query.lower()
        
        # Check if it's a city-specific request (either explicit in query OR when user requested different city)
        if any(city_name in query_lower for city_name in ['noida', 'gurgaon', 'delhi', 'mumbai', 'bangalore', 'pune', 'chennai']):
            return self._generate_no_city_events_message(city, query)
        
        # If no events found for a specific requested city, also use city-specific message
        if city and city.lower() in ['noida', 'gurgaon', 'delhi', 'mumbai', 'bangalore', 'pune', 'chennai']:
            return self._generate_no_city_events_message(city, query)
        
        # Check if it's too vague
        if len(query.split()) <= 2 or query_lower in ['events', 'find events', 'show me events']:
            return self._generate_vague_search_message(city)
        
        # Check for time constraints
        if any(time_word in query_lower for time_word in ['morning', 'evening', 'weekend', 'weekday']):
            time_constraint = next(word for word in ['morning', 'evening', 'weekend', 'weekday'] if word in query_lower)
            return self._generate_no_time_match_message(city, time_constraint)
        
        # Check for budget constraints
        if any(budget_word in query_lower for budget_word in ['free', 'cheap', 'budget', 'under']):
            return self._generate_budget_constraint_message(city, "₹200")
        
        # Default to generic friendly message
        return self._generate_generic_empty_message(city)
    
    def _generate_natural_response(self, events: list, query: str, current_city: str) -> str:
        """Generate natural language response for UI"""
        if not events:
            return f"I couldn't find any events matching your request in {current_city}. Try adjusting your preferences or checking other cities."
        
        response = f"Great news! I found {len(events)} amazing events for you"
        if current_city:
            response += f" in {current_city.title()}"
        response += ":\n\n"
        
        for i, event in enumerate(events[:3], 1):  # Show top 3
            # Get event name with fallbacks
            event_name = event.get('name', '').strip()
            if not event_name or event_name == '****':
                event_name = event.get('event_name', '').strip()
            if not event_name:
                event_name = event.get('activity', 'Event').strip()
            if not event_name:
                event_name = f"Event at {event.get('location', {}).get('venue', 'Venue')}"
            
            # Get location details with fallbacks
            location = event.get('location', {})
            venue = location.get('venue', 'Venue TBD')
            area = location.get('area', 'Area TBD')
            
            response += f"{i}. 🎉 **{event_name}**\n"
            response += f"   📍 {venue}, {area}\n"
            response += f"   ⏰ {event.get('start_time', 'Time TBD')} to {event.get('end_time', 'End TBD')}\n"
            response += f"   💰 ₹{event.get('price', 0)}\n"
            response += f"   🔗 Register: {event.get('registration_url', 'Contact organizer')}\n\n"
        
        if len(events) > 3:
            response += f"...and {len(events) - 3} more great options!"
        
        return response

    def get_bot_response_json(self, user_message: str, user_id: str = None) -> str:
        """Get bot response in JSON-friendly format with user-specific context"""
        try:
            # Add message to user's conversation history
            self._add_to_user_conversation(user_id, "user", user_message)
            
            # Get user-specific context for the AI
            user_history = self._get_user_conversation_history(user_id)
            
            # Create context with user's conversation history
            history_context = "\nConversation History:\n"
            for msg in user_history[-6:]:  # Last 6 messages
                history_context += f"{msg['role']}: {msg['content']}\n"
            
            # Create enhanced message with user context
            if user_id:
                enhanced_message = f"[User ID: {user_id}] {user_message}"
            else:
                enhanced_message = user_message
            
            # For now, use the old method but with enhanced context
            # TODO: Create a proper user-context aware bot response method
            response = self.get_bot_response(enhanced_message)
            
            # Add response to user's conversation history
            clean_response = response.replace("Miffy: ", "").strip()
            self._add_to_user_conversation(user_id, "assistant", clean_response)
            
            return clean_response
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_recommendations_with_json_extraction(self, request_data: dict) -> dict:
        """
        Enhanced version that ensures JSON extraction from AI responses
        """
        try:
            # First get structured recommendations from database
            result = self.get_recommendations_json(request_data)
            
            # Validate and ensure proper JSON structure for each event
            if result.get("recommendations"):  # Check if we have recommendations regardless of success flag
                validated_events = []
                for event in result.get("recommendations", []):
                    # Skip events that have AI reasoning text in the name
                    event_name = str(event.get('name', event.get('event_name', '')))
                    
                    # Filter out events with AI reasoning patterns
                    if any(pattern in event_name.lower() for pattern in [
                        '**', 'not a match', 'exclude', 'skip', 'perfect!', 'wait,',
                        'though', 'might not fit', 'so this', 'so exclude'
                    ]):
                        continue  # Skip this event as it contains AI reasoning
                    
                    # Apply same name fallback logic as in get_recommendations_json
                    if not event_name:
                        event_name = event.get('event_name', '').strip()
                    if not event_name:
                        event_name = f"{event.get('activity', 'Event')} at {event.get('location', {}).get('venue', 'Venue')}"
                    
                    # Only include events with proper data (allow generated names)
                    if not event_name or event_name in ['Unknown']:
                        continue
                    
                    # Ensure all required fields are present
                    validated_event = {
                        "event_id": str(event.get('event_id', '')),
                        "name": event_name,
                        "club_name": str(event.get('club_name', 'Unknown Club')),
                        "activity": str(event.get('activity', 'General')),
                        "start_time": str(event.get('start_time', '')),
                        "end_time": str(event.get('end_time', '')),
                        "location": event.get('location', {}),
                        "price": self._safe_float(event.get('price', event.get('ticket_price', 0))),
                        "available_spots": self._safe_int(event.get('available_spots', 0)),
                        "registration_url": str(event.get('registration_url', 'Contact organizer')),
                        # New fields from API
                        "activity_icon_url": str(event.get('activity_icon_url', '')),
                        "club_icon_url": str(event.get('club_icon_url', '')),
                        "event_cover_image_url": str(event.get('event_cover_image_url', '')),
                        "event_uuid": str(event.get('event_uuid', '')),
                        "participants_count": self._safe_int(event.get('participants_count', 0))
                    }
                    
                    # Ensure location is properly structured
                    if not isinstance(validated_event['location'], dict):
                        validated_event['location'] = {}
                    
                    validated_events.append(validated_event)
                
                # Update result with clean, validated events
                result["recommendations"] = validated_events
                result["total_found"] = len(validated_events)
                
                # If we filtered out all events (they were all AI reasoning), 
                # treat as no results found
                if not validated_events:
                    result["success"] = False
                    result["message"] = result.get("message", "No events found")
                else:
                    result["success"] = True
                
                return result
            
            # If no recommendations from DB, return the result as-is (no AI fallback)
            # This prevents AI reasoning text from being included in responses
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recommendations": [],
                "message": f"Error: {str(e)}"
            }

if __name__ == "__main__":
    # Create bot instance with auto-sync enabled (default)
    # To disable auto-sync, use: bot = MeetupBot(auto_sync=False)
    bot = MeetupBot()

    # Instructions for use
    print("\n🚀 Welcome to Miffy - Your Personal Meetup Recommendation Assistant!")
    print("=" * 60)
    print("📋 Instructions:")
    print("1. Miffy automatically synced events during initialization")
    print("2. Start chatting with Miffy about events you're interested in")
    print("3. Miffy will recommend events based on your preferences")
    print("4. Type 'quit' to exit the conversation")
    print("")
    print("🔥 NEW: JSON API for Recommendations!")
    print("• bot.get_recommendations_json(request_data)")
    print("• Returns structured JSON with match scores and reasons")
    print("• Perfect for mobile apps and web integrations!")
    print("=" * 60)

    # Note: Auto-sync already happened during MeetupBot.__init__()
    # Manual sync is still available: bot.sync_events_once(full_sync=True)
    print(f"\n🔧 ChromaDB server: {bot.chroma_manager.host}:{bot.chroma_manager.port}")
    time.sleep(1)

    # Check event count after sync
    try:
        collection_info = bot.chroma_manager.get_collection_stats()
        existing_events = collection_info.get('total_events', 0)
        if existing_events > 0:
            print(f"✅ ChromaDB now has {existing_events} events!")
        else:
            print("❌ No events found after sync. Please check API connection.")
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
        
       #  try:
       #      tmp_mgr = bot.user_pref_sync_manager
       #    #   imported = tmp_mgr.prompt_and_import_csv_interactive()
       #      if imported > 0:
       #          print(f"✅ Imported {imported} user preferences from CSV", flush=True)
       #          user_prefs_info = bot.chroma_manager.get_user_prefs_stats()
       #          existing_user_prefs = user_prefs_info.get('total_user_preferences', 0)
       #          print(f"🔧 Debug: Total user preferences count after CSV import: {existing_user_prefs}")
       #      else:
       #          print("ℹ️ No CSV import performed.")
       #  except Exception as e:
       #      print(f"❌ CSV import attempt failed: {e}", flush=True)
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
        print("\nNow uses upsert to preserve existing data and add new events!")
        print("Periodic sync focuses on '/updated' API for efficient updates every 2 minutes.")
        print("\n📞 API Call Pattern:")
        print("• Server START (always) → /upcoming + /updated APIs (ensures fresh upcoming events)")
        print("• Every 2 minutes → /updated API only (incremental sync for changes)")
        print("• Manual full sync → /upcoming + /updated APIs (complete refresh)")
        print("• All syncs use UPSERT → preserves old events, updates existing, adds new")
         
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