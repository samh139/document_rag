# project_root/frontend/streamlit_app.py

import streamlit as st
import requests
import uuid

# -------------------------------
# Configuration
# -------------------------------
API_URL = "http://localhost:8000/api/chat"

st.set_page_config(page_title="Banking RAG Assistant", layout="wide")

# -------------------------------
# Session state
# -------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess-{uuid.uuid4().hex[:8]}"

if "user_id" not in st.session_state:
    st.session_state.user_id = "user-123"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# UI
# -------------------------------
st.title("🏦 Banking RAG Assistant")
st.caption("Ask banking-related questions based on RBI documents")

user_input = st.text_input("Ask your question:")

# -------------------------------
# Send message
# -------------------------------
if st.button("Send") and user_input.strip():
    payload = {
        "message": user_input,
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.session_id,
    }

    with st.spinner("Thinking..."):
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

    # Store chat
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": data
    })

# -------------------------------
# Display chat
# -------------------------------
for turn in st.session_state.chat_history:
    st.markdown("### 👤 You")
    st.write(turn["user"])

    st.markdown("### 🤖 Assistant")
    st.write(turn["bot"]["answer"])

    # Citations (optional)
    citations = turn["bot"].get("citations", [])
    if citations:
        with st.expander("📚 Sources"):
            for i, c in enumerate(citations, start=1):
                st.markdown(f"**{i}. {c['file_name']}**")
                st.code(c["chunk_content"])
