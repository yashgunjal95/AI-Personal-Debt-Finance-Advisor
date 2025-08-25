# core/education.py
import os
from typing import Tuple, List
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
GROQ_LLM_MODEL = os.getenv("LLM_MODEL", "Gemma2-9b-It")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

SYNTHESIS_SYSTEM_PROMPT = """You are a helpful, concise finance tutor who must base answers on the provided context.
- Answer directly and clearly.
- When relevant, give short actionable steps, simple numeric examples, and clear definitions.
- At the end, include a "Sources:" section listing the filenames or doc titles used from the context.
- If the context doesn't contain enough info to answer confidently, say so and suggest what extra info is needed.
"""

def _ollama_embedding():
    try:
        return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL)
    except Exception as e:
        raise RuntimeError(f"Ollama embeddings initialization failed: {e}")

@st.cache_resource(show_spinner=False)
def get_or_build_vectorstore(kb_dir: str = "kb") -> FAISS:
    emb = _ollama_embedding()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for root, _, files in os.walk(kb_dir):
        for fname in sorted(files):
            if not fname.lower().endswith((".md", ".txt")):
                continue
            full = os.path.join(root, fname)
            try:
                loader = TextLoader(full, encoding="utf-8")
                loaded = loader.load()
                for d in loaded:
                    if not d.metadata:
                        d.metadata = {}
                    d.metadata["source"] = fname
                splits = splitter.split_documents(loaded)
                docs.extend(splits)
            except Exception:
                pass
    if not docs:
        empty_vs = FAISS.from_texts([""], emb)
        return empty_vs
    vs = FAISS.from_documents(docs, emb)
    return vs

def _get_llm():
    return ChatGroq(model=GROQ_LLM_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.2)

def rag_answer(query: str, vs: FAISS, k: int = 4) -> Tuple[str, List[str]]:
    try:
        docs = vs.similarity_search(query, k=k)
    except Exception as e:
        return f"(Retrieval error) {e}", []
    if not docs:
        return "I couldn't find any documents relevant to that question in the KB.", []
    context_pieces = []
    sources = []
    for i, d in enumerate(docs, start=1):
        content = d.page_content.strip()
        src = d.metadata.get("source", f"kb_{i}")
        sources.append(src)
        context_pieces.append(f"---\nSource: {src}\n\n{content}\n")
    combined_context = "\n\n".join(context_pieces)
    system = SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT)
    human = HumanMessage(content=f"Context:\n\n{combined_context}\n\nQuestion: {query}\n\nAnswer concisely and include a 'Sources:' line.")
    llm = _get_llm()
    try:
        resp = llm.invoke([system, human])
        answer_text = resp.content
    except Exception as e:
        preview = combined_context[:2000]
        return f"(LLM error) {e}\n\nContext preview:\n\n{preview}", sources
    unique_sources = []
    for s in sources:
        if s not in unique_sources:
            unique_sources.append(s)
    return answer_text, unique_sources
