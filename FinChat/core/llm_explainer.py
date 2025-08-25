# core/llm_explainer.py
import os
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
GROQ_MODEL = os.getenv("LLM_MODEL", "Gemma2-9b-It")

SYSTEM = """You are a helpful, concise financial assistant. Given a short recommendation record and user's profile, generate a clear one-paragraph explanation in plain language, include the estimated savings if present, and 2 short next-step actions."""

def explain_recommendation_with_llm(rec: Dict[str, Any], profile: Dict[str, Any], debts: list) -> str:
    if not GROQ_API_KEY:
        return rec.get("explanation", "")
    try:
        llm = ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.2)
        sys = SystemMessage(content=SYSTEM)
        human = HumanMessage(content=f"Recommendation: {rec}\n\nProfile: {profile}\n\nDebts: {debts}\n\nPlease produce a short explanation and 2 concrete next steps.")
        resp = llm.invoke([sys, human])
        return resp.content
    except Exception as e:
        return rec.get("explanation", f"(LLM error: {e})")
