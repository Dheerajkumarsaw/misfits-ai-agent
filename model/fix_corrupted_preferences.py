#!/usr/bin/env python3
"""
Fix corrupted user preference data in ChromaDB
Extracts activity names from full club records
"""
import sys
sys.path.insert(0, '.')
from ai_agent import MeetupBot

print("=" * 70)
print("🔧 Fixing Corrupted User Preference Data")
print("=" * 70)

# Initialize bot
bot = MeetupBot(auto_sync=False)

# Users with corrupted data
corrupted_users = ["1", "452"]

for user_id in corrupted_users:
    print(f"\n📋 Processing user {user_id}...")

    # Get current preferences
    prefs = bot.chroma_manager.get_user_preferences_by_user_id(user_id)

    if not prefs:
        print(f"  ⚠️  No preferences found for user {user_id}")
        continue

    # Get corrupted activities_summary
    metadata = prefs.get('metadata', {})
    activities_summary = metadata.get('activities_summary', '')

    print(f"  📌 Current (corrupted): {activities_summary[:100]}...")

    # Parse corrupted format: "CLUB|ACTIVITY|CITY|AREA|COUNT ; CLUB2|ACTIVITY2..."
    activities = []
    parts = activities_summary.split(' ; ')
    for part in parts:
        fields = part.split('|')
        if len(fields) >= 2:
            activity = fields[1].strip().upper()  # Extract ACTIVITY field
            if activity and activity not in activities:
                activities.append(activity)

    if not activities:
        print(f"  ⚠️  Could not extract activities from corrupted data")
        continue

    print(f"  ✅ Extracted activities: {activities}")

    # Create clean preference data
    clean_pref = {
        "user_id": user_id,
        "activities": activities,
        "preferred_locations": [prefs.get('current_city', 'Gurgaon')],
        "preferred_time": None,
        "budget_range": None,
        "created_at": metadata.get('created_at', ''),
        "activities_summary": ", ".join(activities)  # Clean format
    }

    # Save clean data (will replace corrupted data)
    success = bot.chroma_manager.add_user_preferences_batch([clean_pref])

    if success:
        print(f"  ✅ Fixed preferences for user {user_id}")

        # Verify fix
        updated_prefs = bot.chroma_manager.get_user_preferences_by_user_id(user_id)
        new_summary = updated_prefs.get('metadata', {}).get('activities_summary', '')
        print(f"  ✅ New (clean): {new_summary}")
    else:
        print(f"  ❌ Failed to update preferences for user {user_id}")

print("\n" + "=" * 70)
print("✅ Cleanup complete!")
print("=" * 70)
