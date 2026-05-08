import streamlit as st
from model import predict
import time

def stream_response(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)
        
# --- Page Config ---
st.set_page_config(page_title="Customer Support Bot", page_icon="🤖")
st.title("🤖 Customer Support Bot")

# --- Common Problems ---
common_problems = [
    "My payment failed",
    "I want to exchange my product",
    "Where is my order",
    "I want a refund",
    "Website is not working",
    "Other (type your issue)"
]

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "bot",
        "content": "👋 Hello! I'm your customer support assistant. Please choose your problem below or type your own concern."
    })

# --- Display Chat History ---
for msg in st.session_state.messages:
    if msg["role"] == "bot":
        with st.chat_message("assistant"):
            st.write_stream(stream_response(msg["content"]))
    else:
        st.chat_message("user").write(msg["content"])

# --- Common Problem Buttons ---
if "selected" not in st.session_state:
    st.session_state.selected = None

cols = st.columns(3)
for i, problem in enumerate(common_problems):
    if cols[i % 3].button(problem):
        st.session_state.selected = problem

# --- Handle Selection ---
if st.session_state.selected:
    selected = st.session_state.selected

    if selected == "Other (type your issue)":
        user_input = st.text_input("Type your concern:")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = predict(user_input)
            st.session_state.messages.append({"role": "bot", "content": response})
            st.session_state.selected = None
            st.rerun()
    else:
        if not any(m["content"] == selected for m in st.session_state.messages):
            st.session_state.messages.append({"role": "user", "content": selected})
            response = predict(selected)
            st.session_state.messages.append({"role": "bot", "content": response})
            st.rerun()