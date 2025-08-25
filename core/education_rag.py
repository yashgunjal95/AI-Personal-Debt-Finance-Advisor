# core/education_rag.py
import os
from typing import List, Tuple, Optional
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler

GROQ_LLM_MODEL = os.getenv("LLM_MODEL", "Gemma2-9b-It")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

class StreamlitCallback(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self._acc = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._acc += token
        try:
            self.placeholder.markdown(self._acc)
        except Exception:
            self.placeholder.write(self._acc)

def _get_llm():
    return ChatGroq(model=GROQ_LLM_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.2)

def build_retrieval_qa_chain(vs: FAISS, chain_type: str = "stuff", k: int = 4) -> RetrievalQA:
    llm = _get_llm()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever, return_source_documents=True)
    return qa

def rag_answer_stream(query: str, vs: FAISS, placeholder: Optional[st.delta_generator] = None, k: int = 4) -> Tuple[str, List[str]]:
    try:
        docs = vs.similarity_search(query, k=k)
    except Exception as e:
        return f"(Retrieval error) {e}", []
    if not docs:
        return "No relevant documents found in the knowledge base.", []
    sources = []
    for d in docs:
        s = d.metadata.get("source", None) or d.metadata.get("source_id", None) or "kb"
        if s not in sources:
            sources.append(s)
    qa_chain = build_retrieval_qa_chain(vs, chain_type="stuff", k=k)
    if placeholder is None:
        placeholder = st.empty()
    cb = StreamlitCallback(placeholder)
    answer_text = ""
    try:
        result = qa_chain({"query": query}, callbacks=[cb])
        if isinstance(result, dict):
            answer_text = result.get("result") or result.get("answer") or str(result)
            src_docs = result.get("source_documents") or []
            for d in src_docs:
                s = getattr(d, "metadata", {}).get("source", None) or getattr(d, "metadata", {}).get("source_id", None)
                if s and s not in sources:
                    sources.append(s)
        else:
            answer_text = str(result)
    except TypeError:
        # fallback to non-streaming run
        try:
            answer_text = qa_chain.run(query)
        except Exception as e:
            return f"(LLM/chain error) {e}", sources
    try:
        placeholder.markdown(answer_text)
    except Exception:
        placeholder.write(answer_text)
    return answer_text, sources
