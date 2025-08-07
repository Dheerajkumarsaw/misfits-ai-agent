# AI Meetup Recommendation Bot with ChromaDB Integration

This project implements an intelligent meetup recommendation system that uses ChromaDB for vector storage and semantic search capabilities.

## 🚀 Features

- **Vector-based Event Search**: Uses ChromaDB for semantic similarity search
- **Category-based Filtering**: Search events by category (Sports, Chill Out, Brainy, Arts)
- **Location-based Search**: Find events by specific locations
- **Conversational AI**: Powered by NVIDIA's API for natural language processing
- **Persistent Storage**: ChromaDB persists data between sessions

## 📁 Project Structure

```
Ai Agents/
├── ai-agent.py              # Main AI agent with ChromaDB integration
├── chroma_db_manager.py     # ChromaDB operations manager
├── test_chroma_integration.py # Test script for ChromaDB functionality
├── requirements.txt          # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install ChromaDB**:
   ```bash
   pip install chromadb
   ```

## 🔧 Usage

### 1. Running the Main AI Agent

```python
# Run the main agent
python ai-agent.py
```

The agent will:
- Prompt you to upload a CSV file with events data
- Automatically add events to ChromaDB for vector search
- Start an interactive conversation for event recommendations

### 2. Testing ChromaDB Integration

```python
# Test the ChromaDB functionality
python test_chroma_integration.py
```

This will:
- Create sample events data
- Test vector search capabilities
- Demonstrate category and location-based searches

### 3. Using ChromaDB Manager Directly

```python
from chroma_db_manager import ChromaDBManager

# Initialize manager
chroma_manager = ChromaDBManager()

# Add events to database
chroma_manager.add_events_to_db(your_events_dataframe)

# Search events
results = chroma_manager.search_events("football", n_results=5)
```

## 📊 Expected CSV Format

Your events CSV should have these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `event_name` | Name of the event | "Pickleball - Beginner Level" |
| `club_name` | Organizing club | "Young Picklers Learners Club" |
| `date_time` | Event date and time | "18 Jul, 8 PM" |
| `location` | Event location | "Blue Court, Sec-65" |
| `charges` | Event pricing | "Rs. 400/- Can pay after meetup" |
| `vibe` | Event description | "Beginner Friendly, Instructional" |
| `category` | Event category | "Sports", "Chill Out", "Brainy", "Arts" |
| `registration_link` | Registration URL | "https://example.com/register" |

## 🔍 Search Capabilities

### 1. Semantic Search
```python
# Find events using natural language
events = chroma_manager.search_events("I want to play sports", n_results=5)
```

### 2. Category Search
```python
# Find events by category
sports_events = chroma_manager.search_by_category("Sports", n_results=10)
brainy_events = chroma_manager.search_by_category("Brainy", n_results=10)
```

### 3. Location Search
```python
# Find events by location
location_events = chroma_manager.search_by_location("Sec-23", n_results=10)
```

## 🗄️ ChromaDB Configuration

### Persistence
- Data is stored in `./chroma_db/` directory
- Automatically persists between sessions
- Collection name: `meetup_events`

### Vector Embeddings
- Uses ChromaDB's default embedding model
- Converts event data to searchable text format
- Supports semantic similarity search

## 🎯 Benefits of ChromaDB Integration

1. **Semantic Search**: Find events using natural language queries
2. **Scalability**: Handles large datasets efficiently
3. **Persistence**: Data survives application restarts
4. **Fast Retrieval**: Vector-based search is much faster than text matching
5. **Flexibility**: Easy to extend with additional search criteria

## 🔧 Customization

### Adding New Search Methods

```python
def search_by_price_range(self, min_price: int, max_price: int):
    """Search events by price range"""
    # Implementation here
    pass
```

### Modifying Event Text Format

```python
def prepare_event_text(self, event_row: pd.Series) -> str:
    """Customize how event data is converted to searchable text"""
    # Your custom format here
    pass
```

## 🐛 Troubleshooting

### Common Issues

1. **ChromaDB Connection Error**:
   - Ensure ChromaDB is properly installed
   - Check if the persist directory is writable

2. **Import Errors**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Search Not Working**:
   - Verify events were added to ChromaDB successfully
   - Check collection info: `chroma_manager.get_collection_info()`

## 📈 Performance Tips

1. **Batch Operations**: Add multiple events at once for better performance
2. **Index Optimization**: ChromaDB automatically optimizes for search queries
3. **Memory Management**: Large datasets may require more memory allocation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test with `test_chroma_integration.py`
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This integration keeps ChromaDB operations in a separate file (`chroma_db_manager.py`) for better code organization, maintainability, and reusability. The main AI agent (`ai-agent.py`) imports and uses the ChromaDB manager as needed. 