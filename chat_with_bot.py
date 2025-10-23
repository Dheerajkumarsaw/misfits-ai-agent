#!/usr/bin/env python3
"""
Simple CLI chat interface to talk with the emotionally intelligent bot
"""
import requests
import json
import sys

API_URL = "http://localhost:8000/api/chat"

def chat():
    print("=" * 70)
    print("🤖 Misfits AI Agent - Chat Interface")
    print("=" * 70)
    print("Talk to the emotionally intelligent, mobile-optimized chatbot!")
    print("Commands: 'quit' or 'exit' to stop")
    print("=" * 70)
    print()

    # Get user info
    user_id = input("Enter your user ID (or press Enter to skip): ").strip()
    if not user_id:
        user_id = "demo_user_123"

    city = input("Enter your city (e.g., Mumbai, Delhi): ").strip()
    if not city:
        city = "Mumbai"

    print()
    print(f"✅ Chatting as User ID: {user_id} from {city}")
    print("=" * 70)
    print()

    while True:
        # Get user message
        user_message = input("You: ").strip()

        if not user_message:
            continue

        if user_message.lower() in ['quit', 'exit', 'bye']:
            print("\n👋 Thanks for chatting! See you next time!")
            break

        # Send to API
        try:
            response = requests.post(
                API_URL,
                json={
                    "message": user_message,
                    "user_id": user_id,
                    "user_current_city": city
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Print bot response
                print(f"\n🤖 Miffy: {data.get('message', 'No response')}\n")

                # Show events if any
                events = data.get('events', [])
                if events:
                    print(f"📋 Found {len(events)} event(s):\n")
                    for i, event in enumerate(events, 1):
                        print(f"{i}. {event.get('name', 'Event')}")
                        location = event.get('location', {})
                        print(f"   📍 {location.get('area', 'N/A')}, {location.get('city', 'N/A')}")
                        print(f"   💰 ₹{event.get('price', 'N/A')}")
                        print(f"   🔗 {event.get('registration_url', 'N/A')}")
                        print()

            else:
                print(f"\n❌ Error: {response.status_code}")
                print(f"Response: {response.text}\n")

        except requests.exceptions.ConnectionError:
            print("\n❌ Cannot connect to server. Is it running?")
            print("Start server with: cd model && python3 api_server.py\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
