#!/usr/bin/env python3
"""
Test to demonstrate smart LLM vs dumb keyword matching
"""
import sys
from graph_tools import LLMTool
from openai import OpenAI
import httpx

# Initialize client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
    http_client=httpx.Client(timeout=30.0)
)

llm_tool = LLMTool(client)

print("=" * 70)
print("🧠 Testing Smart LLM Intent Detection")
print("=" * 70)

# Test cases that show LLM intelligence vs keyword matching
test_cases = [
    {
        "query": "best for me",
        "has_events": False,
        "expected": "new_search",
        "why": "User wants best events but has no results yet"
    },
    {
        "query": "best for me",
        "has_events": True,
        "expected": "best_picks",
        "why": "User wants best from existing results"
    },
    {
        "query": "what's happening tonight",
        "has_events": False,
        "expected": "new_search",
        "why": "Time-based search query"
    },
    {
        "query": "I'm looking for something fun this weekend",
        "has_events": False,
        "expected": "new_search",
        "why": "Natural language event request"
    },
    {
        "query": "hi there",
        "has_events": False,
        "expected": "greeting",
        "why": "Short greeting"
    },
    {
        "query": "hi, show me football events",
        "has_events": False,
        "expected": "new_search",
        "why": "Greeting + search (search wins)"
    },
    {
        "query": "actually, show me the perfect match",
        "has_events": True,
        "expected": "best_picks",
        "why": "Conversational continuation with 'perfect' keyword"
    },
    {
        "query": "any music concerts nearby?",
        "has_events": False,
        "expected": "new_search",
        "why": "Question format for event search"
    },
]

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    print(f"\n{'=' * 70}")
    print(f"Test {i}/{len(test_cases)}")
    print(f"Query: \"{test['query']}\"")
    print(f"Has events in session: {test['has_events']}")
    print(f"Expected: {test['expected']} ({test['why']})")

    result = llm_tool.detect_intent(
        query=test['query'],
        conversation_history=[],
        has_events_in_session=test['has_events']
    )

    detected = result.get('intent')
    confidence = result.get('confidence', 0)
    reasoning = result.get('reasoning', '')

    print(f"Detected: {detected} (confidence: {confidence:.2f})")
    print(f"Reasoning: {reasoning}")

    if detected == test['expected']:
        print("✅ PASS")
        passed += 1
    else:
        print(f"❌ FAIL (expected {test['expected']}, got {detected})")
        failed += 1

print(f"\n{'=' * 70}")
print(f"📊 Results: {passed}/{len(test_cases)} passed ({(passed/len(test_cases)*100):.1f}%)")
print(f"{'=' * 70}")

if failed == 0:
    print("🎉 All tests passed! LLM is working smartly!")
else:
    print(f"⚠️  {failed} tests failed - may need prompt tuning")
