import chromadb
from chromadb.config import Settings
from embedding_utils import get_embedding

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_data"))
collection = client.get_or_create_collection(name="events")

def build_text(event):
    return f"{event['title']} in {event['location']} on {event['datetime']} for {event['price']} INR - Hobby: {event['hobby']}"

def create_or_update_event(event):
    doc_id = str(event['event_id'])
    text = build_text(event)
    embedding = get_embedding(text)

    try:
        # Try to update (if exists)
        collection.update(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[event],
        )
        print(f"Updated event: {doc_id}")
    except:
        # Fallback to add (new event)
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[event],
        )
        print(f"Created new event: {doc_id}")

def query_similar_events(query_text, top_k=5):
    embedding = get_embedding(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results
