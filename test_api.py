"""
Test script for the Meetup Bot API
"""

import requests
import json

# API endpoint (change this to your deployed URL)
API_URL = "http://localhost:8000"

def test_recommendations():
    """Test the recommendations endpoint"""
    
    # Example 1: User in Gurugram looking for events
    request1 = {
        "query": "I want to play cricket this weekend",
        "user_id": "user123",
        "user_current_city": "gurugram",
        "limit": 5
    }
    
    print("=" * 60)
    print("TEST 1: User in Gurugram looking for cricket")
    print("Request:", json.dumps(request1, indent=2))
    
    response = requests.post(f"{API_URL}/api/recommendations", json=request1)
    result = response.json()
    
    print("\nResponse Summary:")
    print(f"Success: {result.get('success')}")
    print(f"Total Found: {result.get('total_found')}")
    print(f"Message: {result.get('message')}")
    
    if result.get('recommendations'):
        print("\nTop Recommendations:")
        for i, event in enumerate(result['recommendations'][:3], 1):
            print(f"\n{i}. {event['name']}")
            print(f"   Location: {event['location']['full_address']}")
            print(f"   Time: {event['start_time']} to {event['end_time']}")
            print(f"   Match Score: {event['match_score']}")
            print(f"   Why: {event['why_recommended']}")
            print(f"   Registration: {event['registration_url']}")
    
    print("\n" + "=" * 60)
    
    # Example 2: User explicitly asking for events in another city
    request2 = {
        "query": "Show me badminton events in Mumbai",
        "user_id": "user456",
        "user_current_city": "gurugram",
        "limit": 3
    }
    
    print("\nTEST 2: User in Gurugram but asking for Mumbai events")
    print("Request:", json.dumps(request2, indent=2))
    
    response = requests.post(f"{API_URL}/api/recommendations", json=request2)
    result = response.json()
    
    print("\nResponse Summary:")
    print(f"Success: {result.get('success')}")
    print(f"Events shown from: {result['recommendations'][0]['location']['city'] if result.get('recommendations') else 'No events'}")
    
    print("\n" + "=" * 60)
    
    # Example 3: Chat endpoint test
    request3 = {
        "query": "I'm new to the city, what fun activities are happening today?",
        "user_id": "newuser",
        "user_current_city": "delhi"
    }
    
    print("\nTEST 3: Chat endpoint with natural language")
    print("Request:", json.dumps(request3, indent=2))
    
    response = requests.post(f"{API_URL}/api/chat", json=request3)
    result = response.json()
    
    if result.get('success'):
        print("\nBot Response:")
        print(result.get('chat_response', 'No response'))
        
        if result.get('combined_response', {}).get('events'):
            print("\nStructured Events:")
            for event in result['combined_response']['events']:
                print(f"- {event['name']} ({event['match_score']})")

def test_health():
    """Test health endpoints"""
    print("\n" + "=" * 60)
    print("HEALTH CHECK")
    
    # Liveness probe
    response = requests.get(f"{API_URL}/health/liveness")
    print(f"Liveness: {response.json()}")
    
    # Readiness probe
    response = requests.get(f"{API_URL}/health/readiness")
    print(f"Readiness: {response.json()}")
    
    # Stats
    response = requests.get(f"{API_URL}/api/stats")
    stats = response.json()
    print(f"\nSystem Stats:")
    print(f"Total Events: {stats.get('events', {}).get('total_events', 0)}")
    print(f"Total User Preferences: {stats.get('user_preferences', {}).get('total_user_preferences', 0)}")

if __name__ == "__main__":
    print("🚀 Testing Meetup Bot API")
    print("=" * 60)
    
    try:
        # Test health first
        test_health()
        
        # Test recommendations
        test_recommendations()
        
        print("\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on", API_URL)
    except Exception as e:
        print(f"❌ Error: {e}")