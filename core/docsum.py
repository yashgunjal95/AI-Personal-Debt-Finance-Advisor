# core/docsum.py
import io
import pdfplumber
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM = "You are a concise financial document summarizer. Extract key numbers, dates, and action items."

def _extract_text(file) -> str:
    name = getattr(file, "name", "").lower()
    if name.endswith(".pdf"):
        try:
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                text = []
                for p in pdf.pages:
                    text.append(p.extract_text() or "")
            return "\n".join(text)
        except Exception as e:
            return f"(PDF read error: {e})"
    elif name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            return df.to_csv(index=False)
        except Exception as e:
            return f"(CSV read error: {e})"
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def summarize_docs(files, llm) -> str:
    texts = []
    for f in files:
        try:
            texts.append(_extract_text(f))
        except Exception as e:
            texts.append(f"Error reading {getattr(f,'name','file')}: {e}")
    all_text = "\n\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(all_text)
    if not chunks:
        return "(No text extracted)"
    partials = []
    for ch in chunks:
        msg = [SystemMessage(content=SYSTEM), HumanMessage(content=f"Summarize this part:\n\n{ch}")]
        try:
            resp = llm.invoke(msg)
            partials.append(resp.content)
        except Exception as e:
            partials.append(f"(LLM error) {e}")
    final_msg = [SystemMessage(content=SYSTEM), HumanMessage(content="Combine these partial summaries into one concise summary:\n\n" + "\n\n".join(partials))]
    try:
        final = llm.invoke(final_msg).content
    except Exception as e:
        final = f"(LLM error combining) {e}"
    return final
