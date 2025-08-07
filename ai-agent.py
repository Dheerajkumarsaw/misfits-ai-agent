# Interactive Meetup Recommendation Bot with ChromaDB Integration
# Install required packages at the start
from pickle import FALSE
import subprocess
import sys

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
packages = ["openai", "pandas", "chromadb", "numpy", "typing-extensions"]
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

# Initialize the NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

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
        print(f"🔧 Initializing ChromaDB at {self.persist_directory}")
        print(f"🔧 Looking for collection: {self.collection_name}")
        
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
        if 'registration_url' in event_row:
            text_parts.append(f"Registration URL: {event_row['registration_url']}")
        
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
                try:
                    existing_items = self.collection.get()
                    ids_to_delete = existing_items.get("ids", [])
                    print(ids_to_delete)
                    if ids_to_delete:
                        self.collection.delete(ids=ids_to_delete)
                        print(f"🗑️ Deleted {len(ids_to_delete)} old events from ChromaDB")
                    else:
                        print("ℹ️ No existing items to delete in ChromaDB.")
                except Exception as e:
                    print(f"❌ Error while clearing ChromaDB collection: {e}")
                    return False
            
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
                    'registration_url': str(row.get('registration_url', '')),
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
            category: Event category (Sports, Arts, Social, etc.)
            n_results: Number of results to return
            
        Returns:
            List of event dictionaries
        """
        try:
            results = self.collection.query(
                query_texts=[category],
                where={"activity": category},
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
                    if location.lower() in metadata.get('location_name', '').lower():
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

class MeetupBot:
    def __init__(self):
        self.events_data = None
        self.conversation_history = []
        self.chroma_manager = ChromaDBManager()
        
    def upload_dataset(self):
        """Upload and load the events dataset"""
        print("Please upload your events dataset (CSV file):")
        uploaded = files.upload()
        for filename in uploaded.keys():
            try:
                # Read the CSV file
                content = uploaded[filename]
                self.events_data = pd.read_csv(io.BytesIO(content))
                print(f"✅ Dataset loaded successfully! Shape: {self.events_data.shape}")
                print(f"Columns: {list(self.events_data.columns)}")
                print("\nFirst few rows:")
                print(self.events_data.head())
                
                # Add events to ChromaDB
                print("\n🔄 Adding events to ChromaDB for vector search...")
                if self.chroma_manager.add_events_to_db(self.events_data):
                    print("✅ Events successfully added to ChromaDB!")
                    # Show collection info
                    info = self.chroma_manager.get_collection_info()
                    print(f"📊 ChromaDB Collection Info: {info}")
                else:
                    print("⚠️ Warning: Failed to add events to ChromaDB, using fallback mode")
                
                return True
            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
                return False
    
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
            events = self.chroma_manager.search_by_category(category, n_results)
            return events
        except Exception as e:
            print(f"❌ Category search error: {e}")
            return []
    
    def search_by_location_vector(self, location: str, n_results: int = 10):
        """Search events by location using ChromaDB"""
        try:
            events = self.chroma_manager.search_by_location(location, n_results)
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
            response += f"**{event.get('name', 'Event')}**\n"
            response += f"1. **Activity - {event.get('activity', 'N/A')}**\n"
            response += f"2. **Date & Time - {event.get('start_time', 'N/A')} to {event.get('end_time', 'N/A')}**\n"
            response += f"3. **Location - {event.get('location_name', 'N/A')}**\n"
            response += f"4. **Area - {event.get('area_name', 'N/A')}, {event.get('city_name', 'N/A')}**\n"
            response += f"5. **Price - Rs. {event.get('ticket_price', 'N/A')}**\n"
            response += f"6. **Available Spots - {event.get('available_spots', 'N/A')}**\n"
            response += f"7. **Description - {event.get('description', 'N/A')}**\n"
            
            if event.get('registration_url'):
                response += f"8. **Register: {event.get('registration_url')}**\n"
            
            response += "\n" + "="*50 + "\n\n"
        
        return response
    
    def prepare_context(self, user_message):
        """Prepare context with dataset information for the AI model"""
        # Check if we have data in ChromaDB even if events_data is None
        collection_info = self.chroma_manager.get_collection_info()
        if self.events_data is None and collection_info.get('total_events', 0) == 0:
            return "No events dataset available."
        
        # Use ChromaDB for semantic search
        relevant_events = self.search_events_vector(user_message, n_results=10)
        
        # Convert relevant events to string format for context
        events_context = "Relevant Events Data (from vector search):\n"
        if relevant_events:
            for event in relevant_events:
                events_context += f"Event: {event.get('name', 'N/A')} | "
                events_context += f"Activity: {event.get('activity', 'N/A')} | "
                events_context += f"Date: {event.get('start_time', 'N/A')} | "
                events_context += f"Location: {event.get('location_name', 'N/A')} | "
                events_context += f"Area: {event.get('area_name', 'N/A')} | "
                events_context += f"City: {event.get('city_name', 'N/A')} | "
                events_context += f"Price: Rs. {event.get('ticket_price', 'N/A')} | "
                events_context += f"Available Spots: {event.get('available_spots', 'N/A')}\n"
        else:
            events_context += "No specific events found for this query.\n"
        
        # Add conversation history
        history_context = "\nConversation History:\n"
        for msg in self.conversation_history[-6:]:  # Keep last 6 messages for context
            history_context += f"{msg['role']}: {msg['content']}\n"
        
        system_prompt = f"""You are a friendly meetup recommendation bot with vector search capabilities. Your role is to help users find events and activities based on their preferences using semantic search.

{events_context}
{history_context}

Instructions:
1. Analyze the user's request and use the relevant events from vector search
2. Categorize events by activity type (Sports, Arts, Social, etc.)
3. Always respond in a conversational, friendly tone
4. When showing events, use this exact format:
   **Event Name**
   1. **Activity - [Activity Type]**
   2. **Date & Time - [Start Time] to [End Time]**
   3. **Location - [Location Name]**
   4. **Area - [Area Name], [City Name]**
   5. **Price - Rs. [Ticket Price]**
   6. **Available Spots - [Available Spots]**
   7. **Description - [Event Description]**
5. If no events match, offer alternatives (different locations, dates, activities)
6. When user selects an event, provide registration link: "Oh great, you can register for the meetup here -> <Registration URL>"
7. Keep responses helpful and engaging
8. Use the vector search results to provide more accurate recommendations
9. Mention availability and pricing clearly

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
        # Check if we have data in ChromaDB even if events_data is None
        collection_info = self.chroma_manager.get_collection_info()
        if self.events_data is None and collection_info.get('total_events', 0) == 0:
            print("❌ No event data available. Please upload a dataset first!")
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

# Create bot instance
bot = MeetupBot()

# Instructions for use
print("🚀 Welcome to the Interactive Meetup Recommendation Bot!")
print("=" * 60)
print("📋 Instructions:")
print("1. First, upload your events dataset (CSV format)")
print("2. Start chatting with the bot about events you're interested in")
print("3. The bot will recommend events based on your preferences")
print("4. Type 'quit' to exit the conversation")
print("=" * 60)

# Step 1: Check if data already exists
print("\n📁 Step 1: Checking for existing data...")
print(f"🔧 Debug: ChromaDB persist directory: {bot.chroma_manager.persist_directory}")

# Add error handling for collection info
try:
    collection_info = bot.chroma_manager.get_collection_info()
    existing_events = collection_info.get('total_events', 0)
    print(f"🔧 Debug: Collection info: {collection_info}")
    print(f"🔧 Debug: Existing events count: {existing_events}")
except Exception as e:
    print(f"🔧 Debug: Error getting collection info: {e}")
    existing_events = 0

if existing_events > 0:
    print(f"✅ Found {existing_events} events in ChromaDB!")
    print("✅ Using existing event data from ChromaDB")
    data_loaded = True
else:
    print("📁 No existing data found. Please upload your events dataset")
    data_loaded = bot.upload_dataset()

if data_loaded:
    print("\n🎉 Great! Dataset ready.")
    # Step 2: Start conversation
    print("\n💬 Step 2: Start chatting with the bot")
    print("Try asking things like:")
    print("- 'Looking for something fun to do today'")
    print("- 'I want to play football'")
    print("- 'Show me sports events'")
    print("- 'Looking to meet new people'")
    bot.start_conversation()
else:
    print("❌ Please fix the dataset upload issue and try again.")