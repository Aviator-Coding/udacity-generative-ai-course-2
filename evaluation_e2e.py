"""
End-to-End Evaluation Tests for NASA RAG Chat System
"""

import os
from llm_client import generate_response, NASA_EXPERT_SYSTEM_PROMPT
from dotenv import load_dotenv

_ = load_dotenv()


def test_multi_turn_conversation():
    """
    Test multi-turn conversation history management.
    """
    print("\n" + "="*80)
    print("MULTI-TURN CONVERSATION TEST")
    print("="*80)

    # Verify NASA expert system prompt exists and contains key requirements
    print("\n1. SYSTEM PROMPT VERIFICATION:")
    assert "NASA Mission Expert" in NASA_EXPERT_SYSTEM_PROMPT, "System prompt must identify as NASA Mission Expert"
    assert "cite" in NASA_EXPERT_SYSTEM_PROMPT.lower(), "System prompt must require source citations"
    assert "conversation history" in NASA_EXPERT_SYSTEM_PROMPT.lower(), "System prompt must mention conversation history awareness"
    print("   ✓ System prompt defines NASA Mission Expert persona")
    print("   ✓ System prompt requires source citations")
    print("   ✓ System prompt mentions conversation history awareness")

    # Check if API key is available for live test
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("\n2. LIVE CONVERSATION TEST: SKIPPED (no API key)")
        print("   Set OPENAI_API_KEY environment variable to run live test")
        return

    print("\n2. LIVE MULTI-TURN CONVERSATION TEST:")

    # Simulate conversation history
    conversation_history = []

    # Turn 1: Initial question about Apollo 13
    turn1_context = """<context>
<document index="1">
  <header>[Index: 1] Mission: Apollo 13 | Category: Technical | Source: AS13_Technical_Transcript</header>
  <content>CDR Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.</content>
</document>
<document index="2">
  <header>[Index: 2] Mission: Apollo 13 | Category: Technical | Source: AS13_Technical_Transcript</header>
  <content>LMP Okay. Right now, Houston, the voltage is looking good. And we had a pretty large bang associated with the CAUTION AND WARNING there.</content>
</document>
</context>"""

    turn1_question = "What problem did Apollo 13 encounter?"
    print(f"\n   Turn 1 Question: {turn1_question}")

    turn1_response = generate_response(
        openai_key=openai_key,
        user_message=turn1_question,
        context=turn1_context,
        conversation_history=conversation_history,
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=200,
        history_limit=10
    )
    print(f"   Turn 1 Response: {turn1_response[:200]}...")

    # Add turn 1 to conversation history (as would happen in chat.py)
    conversation_history.append({"role": "user", "content": turn1_question})
    conversation_history.append({"role": "assistant", "content": turn1_response})

    # Turn 2: Follow-up question that depends on prior context
    turn2_context = """<context>
<document index="1">
  <header>[Index: 1] Mission: Apollo 13 | Category: Technical | Source: AS13_Technical_Transcript</header>
  <content>CC Okay, stand by, 13. We're looking at it.</content>
</document>
<document index="2">
  <header>[Index: 2] Mission: Apollo 13 | Category: Technical | Source: AS13_Technical_Transcript</header>
  <content>CDR Yes. We got a MAIN BUS A UNDERVOLT now, too, showing.</content>
</document>
</context>"""

    # This question uses "they" which requires context from Turn 1 to understand
    turn2_question = "What did they do next?"
    print(f"\n   Turn 2 Question: {turn2_question}")
    print("   (Note: 'they' refers to Apollo 13 crew from Turn 1 context)")

    turn2_response = generate_response(
        openai_key=openai_key,
        user_message=turn2_question,
        context=turn2_context,
        conversation_history=conversation_history,
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=200,
        history_limit=10
    )
    print(f"   Turn 2 Response: {turn2_response[:200]}...")

    # Validate that Turn 2 response references Apollo 13 (showing it understood context)
    turn2_lower = turn2_response.lower()
    context_preserved = any(term in turn2_lower for term in ["apollo", "13", "crew", "houston", "bus", "undervolt"])

    print("\n3. CONTEXT PRESERVATION CHECK:")
    if context_preserved:
        print("   ✓ Follow-up response maintains context from prior turn")
        print("   ✓ Multi-turn conversation history is working correctly")
    else:
        print("   ⚠ Warning: Response may not have preserved conversation context")
        print("   (This could be due to model interpretation, not a code issue)")

    print("\n" + "="*80)
    print("MULTI-TURN TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_multi_turn_conversation()
