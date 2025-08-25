# core/memory.py
import streamlit as st

def get_chat_history(key: str):
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]

def add_chat_message(key: str, role: str, content: str):
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append({"role": role, "content": content})

def reset_chat_history(key: str):
    st.session_state[key] = []
