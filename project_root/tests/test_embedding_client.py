import pytest
import requests
from unittest.mock import patch, MagicMock

from app.agents.rag.embedding_client import embed_text, OLLAMA, EMBED_MODEL


# -------------------------------
# SUCCESS CASE: embedding returned
# -------------------------------
@patch("agents.rag.embedding_client.requests.post")
def test_embed_text_success(mock_post):
    # fake embedding returned by Ollama
    fake_response = {
        "model": EMBED_MODEL,
        "embeddings": [[0.1, 0.2, 0.3]]
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    out = embed_text("hello world")

    # Validate final output vector
    assert isinstance(out, list)
    assert out == [0.1, 0.2, 0.3]

    # Validate HTTP call details
    mock_post.assert_called_once_with(
        f"{OLLAMA}/api/embed",
        json={"model": EMBED_MODEL, "input": ["hello world"]},
        timeout=60
    )


# --------------------------------------
# FAILURE CASE: Ollama returns bad JSON
# --------------------------------------
@patch("agents.rag.embedding_client.requests.post")
def test_embed_text_bad_json(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"unexpected": "format"}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    out = embed_text("test text")

    # fallback returns full JSON body
    assert out == {"unexpected": "format"}


# ---------------------------------------------------
# FAILURE CASE: HTTP request error triggers exception
# ---------------------------------------------------
@patch("agents.rag.embedding_client.requests.post")
def test_embed_text_http_error(mock_post):
    mock_post.side_effect = requests.exceptions.RequestException("connection failed")

    with pytest.raises(requests.exceptions.RequestException):
        embed_text("hello")
