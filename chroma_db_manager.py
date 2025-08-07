import chromadb
from chromadb.config import Settings
import pandas as pd
import json
from typing import List, Dict, Any, Optional
import uuid
import os

class ChromaDBManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB manager for meetup events
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        print(f"💾 Using local directory: {persist_directory}")
        
        # Create directory if it doesn't exist
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            print(f"✅ Created directory: {persist_directory}")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "meetup_events"
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the events collection"""
        print(f"🔧 Debug: Initializing ChromaDB at {self.persist_directory}")
        print(f"🔧 Debug: Looking for collection: {self.collection_name}")
        
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            count = self.collection.count()
            print(f"✅ Connected to existing collection: {self.collection_name} with {count} items")
        except Exception as e:
            print(f"🔧 Debug: Collection not found, creating new one. Error: {e}")
            # Create new collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Meetup events for recommendation system"}
            )
            print(f"✅ Created new collection: {self.collection_name}")
    
    def prepare_event_text(self, event_row: pd.Series) -> str:
        """
        Convert event data to searchable text based on PostgreSQL query structure
        
        Args:
            event_row: Pandas Series containing event data
            
        Returns:
            Formatted text for vector embedding
        """
        text_parts = []
        
        # Add event name
        if 'name' in event_row:
            text_parts.append(f"Event: {event_row['name']}")
        
        # Add description
        if 'description' in event_row:
            text_parts.append(f"Description: {event_row['description']}")
        
        # Add activity
        if 'activity' in event_row:
            text_parts.append(f"Activity: {event_row['activity']}")
        
        # Add date and time
        if 'start_time' in event_row:
            text_parts.append(f"Start Time: {event_row['start_time']}")
        if 'end_time' in event_row:
            text_parts.append(f"End Time: {event_row['end_time']}")
        
        # Add location information
        if 'location_name' in event_row:
            text_parts.append(f"Location: {event_row['location_name']}")
        if 'area_name' in event_row:
            text_parts.append(f"Area: {event_row['area_name']}")
        if 'city_name' in event_row:
            text_parts.append(f"City: {event_row['city_name']}")
        if 'city_state' in event_row:
            text_parts.append(f"State: {event_row['city_state']}")
        
        # Add pricing and availability
        if 'ticket_price' in event_row:
            text_parts.append(f"Price: Rs. {event_row['ticket_price']}")
        if 'available_spots' in event_row:
            text_parts.append(f"Available Spots: {event_row['available_spots']}")
        if 'max_people' in event_row:
            text_parts.append(f"Max Capacity: {event_row['max_people']}")
        
        # Add event URL
        if 'event_url' in event_row:
            text_parts.append(f"Event URL: {event_row['event_url']}")
        
        return " | ".join(text_parts)
    
    def add_events_to_db(self, events_df: pd.DataFrame, clear_existing: bool = True) -> bool:
        """
        Add events from DataFrame to ChromaDB
        
        Args:
            events_df: Pandas DataFrame containing events data
            clear_existing: Whether to clear existing data before adding new events
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear existing data only if requested
            if clear_existing:
                self.collection.delete(where={})
                print("🗑️ Cleared existing data from ChromaDB")
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in events_df.iterrows():
                # Prepare document text
                doc_text = self.prepare_event_text(row)
                
                # Create metadata
                metadata = {
                    'event_id': str(row.get('event_id', '')),
                    'name': str(row.get('name', '')),
                    'description': str(row.get('description', '')),
                    'activity': str(row.get('activity', '')),
                    'start_time': str(row.get('start_time', '')),
                    'end_time': str(row.get('end_time', '')),
                    'allowed_friends': str(row.get('allowed_friends', '')),
                    'ticket_price': str(row.get('ticket_price', '')),
                    'event_url': str(row.get('event_url', '')),
                    'available_spots': str(row.get('available_spots', '')),
                    'max_people': str(row.get('max_people', '')),
                    'location_name': str(row.get('location_name', '')),
                    'location_url': str(row.get('location_url', '')),
                    'area_name': str(row.get('area_name', '')),
                    'city_name': str(row.get('city_name', '')),
                    'city_state': str(row.get('city_state', ''))
                }
                
                # Generate unique ID
                event_id = str(uuid.uuid4())
                
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(event_id)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Added {len(documents)} events to ChromaDB")
            return True
            
        except Exception as e:
            print(f"❌ Error adding events to ChromaDB: {e}")
            return False
    
    def search_events(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for events using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of event dictionaries with metadata
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            events = []
            if results['metadatas'] and results['metadatas'][0]:
                for metadata in results['metadatas'][0]:
                    events.append(metadata)
            
            return events
            
        except Exception as e:
            print(f"❌ Error searching events: {e}")
            return []
    
    def search_by_category(self, category: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search events by category
        
        Args:
            category: Event category (Sports, Chill Out, Brainy, Arts)
            n_results: Number of results to return
            
        Returns:
            List of event dictionaries
        """
        try:
            results = self.collection.query(
                query_texts=[category],
                where={"category": category},
                n_results=n_results
            )
            
            events = []
            if results['metadatas'] and results['metadatas'][0]:
                for metadata in results['metadatas'][0]:
                    events.append(metadata)
            
            return events
            
        except Exception as e:
            print(f"❌ Error searching by category: {e}")
            return []
    
    def search_by_location(self, location: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search events by location
        
        Args:
            location: Location to search for
            n_results: Number of results to return
            
        Returns:
            List of event dictionaries
        """
        try:
            results = self.collection.query(
                query_texts=[location],
                n_results=n_results
            )
            
            events = []
            if results['metadatas'] and results['metadatas'][0]:
                for metadata in results['metadatas'][0]:
                    if location.lower() in metadata.get('location', '').lower():
                        events.append(metadata)
            
            return events
            
        except Exception as e:
            print(f"❌ Error searching by location: {e}")
            return []
    
    def get_all_events(self) -> List[Dict[str, Any]]:
        """
        Get all events from the database
        
        Returns:
            List of all event dictionaries
        """
        try:
            results = self.collection.get()
            
            events = []
            if results['metadatas']:
                for metadata in results['metadatas']:
                    events.append(metadata)
            
            return events
            
        except Exception as e:
            print(f"❌ Error getting all events: {e}")
            return []
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"✅ Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_events": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return {} 
