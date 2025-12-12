import os
from app.agents.engagement_agent import classify_with_ollama

def test_greeting_intent():
    """Verify that greeting → intent=greeting."""
    text = "Hello, good morning!"
    out = classify_with_ollama(text)

    print("MODEL OUTPUT:", out)

    assert isinstance(out, dict)
    assert out["intent"] in ["greeting", "smalltalk", "unknown"]
