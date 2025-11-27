# v5_rag_clean.py
import os
import json
import csv
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# -------- CONFIG --------
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FETCH_K = 20       # how many candidates to pull from vectorstore
TOP_K = 5          # how many after rerank to give to LLM
CONFIDENCE_THRESHOLD = 1.8   # if top score < threshold => fallback
DEBUG = True

# -------- HELPERS --------
def load_service_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Normalize NaNs to empty string for safe string ops
    df = df.fillna("")
    # Ensure numeric columns as strings where appropriate
    df["Giá (VNĐ)"] = df["Giá (VNĐ)"].apply(lambda x: int(x) if str(x).strip() != "" else 0)
    df["Chu kỳ (ngày)"] = df["Chu kỳ (ngày)"].apply(lambda x: int(x) if str(x).strip() != "" else 0)
    return df

def df_to_documents(df: pd.DataFrame, source: str) -> List[Document]:
    docs = []
    for _, row in df.iterrows():
        code = str(row.get("Mã dịch vụ", "")).strip()
        # Build a consistent textual representation (concise)
        text_parts = []
        for col in df.columns:
            val = row[col]
            # make it compact
            text_parts.append(f"{col}: {val}")
        text = " . ".join(text_parts)
        docs.append(Document(page_content=text, metadata={"source": source, "service_code": code}))
    return docs

def chunk_documents(docs: List[Document], chunk_size:int=512, chunk_overlap:int=50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def build_vectorstore(docs: List[Document], persist_dir:str=PERSIST_DIR):
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # Use deterministic ids to avoid duplicates if re-building
    ids = [f"{d.metadata.get('service_code','unknown')}_{i}" for i, d in enumerate(docs)]
    vect = Chroma.from_documents(documents=docs, embedding=emb, ids=ids, persist_directory=persist_dir)
    return vect

# -------- Retriever + Reranker that returns scores too --------
def make_reranked_retriever_with_scores(vectorstore, fetch_k:int=FETCH_K, top_k:int=TOP_K):
    retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    reranker = CrossEncoder(RERANKER_MODEL)

    def retrieve_with_scores(question: str) -> List[Tuple[Document, float]]:
        # get candidates (list of Document)
        candidates = retriever.invoke(question)
        if DEBUG:
            print(f"[DEBUG] raw candidates count: {len(candidates)}")
        # dedupe by service_code (keep first occurrence)
        uniq = []
        seen = set()
        for d in candidates:
            code = d.metadata.get("service_code")
            if code not in seen:
                uniq.append(d)
                seen.add(code)
        candidates = uniq
        if DEBUG:
            print(f"[DEBUG] deduped candidates count: {len(candidates)}")
            for d in candidates[:10]:
                meta = d.metadata
                print("[CAND]", meta, "->", d.page_content[:120].replace("\n"," "))
        if not candidates:
            return []

        pairs = [(question, d.page_content) for d in candidates]
        scores = reranker.predict(pairs)  # array of floats
        scored = list(zip(candidates, [float(s) for s in scores]))
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        if DEBUG:
            print("[DEBUG] top scored (doc, score):")
            for doc, sc in scored_sorted[:min(10, len(scored_sorted))]:
                print(f"  {doc.metadata.get('service_code')} -> {sc:.4f}")
        return scored_sorted[:top_k]

    return retrieve_with_scores

# -------- RAG Prompt (strict) --------
RAG_PROMPT = """
Bạn là một trợ lý hỗ trợ khách hàng về các gói cước (Viettel).
Hãy **chỉ** trả lời dựa trên NGỮ CẢNH (Context) được cung cấp dưới đây.
KHÔNG bổ sung thông tin nào không có trong context. 
Nếu context không chứa câu trả lời rõ ràng, trả lời đúng:
"Tôi không biết - vui lòng liên hệ tổng đài 18001090 hoặc email support@telco.vn"

Khi trả lời, luôn trích dẫn nguồn theo dạng [source: filename].

Context:
{context}

Câu hỏi:
{question}

Trả lời ngắn gọn, bằng tiếng Việt.
"""

prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)

# -------- Make QA chain that uses retriever_with_scores and confidence threshold --------
def make_qa_chain(vectorstore, model_name:str="gemini-2.0-flash", temperature:float=0.0, fetch_k:int=FETCH_K, top_k:int=TOP_K, confidence_threshold:float=CONFIDENCE_THRESHOLD):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    retriever_with_scores = make_reranked_retriever_with_scores(vectorstore, fetch_k=fetch_k, top_k=top_k)

    def answer_query(question: str):
        scored = retriever_with_scores(question)  # List[(Document, score)]
        if not scored:
            if DEBUG: print("[DEBUG] No candidates -> fallback")
            return {"answer": "Tôi không biết - vui lòng liên hệ tổng đài 18001090 hoặc email support@telco.vn", "docs": [], "hallucinated": False}

        docs, scores = zip(*scored)
        # Confidence check: use top score (or average) to decide
        top_score = float(scores[0])
        avg_score = float(sum(scores)/len(scores))
        if DEBUG:
            print(f"[DEBUG] top_score={top_score:.4f}, avg_score={avg_score:.4f}, threshold={confidence_threshold}")

        if top_score < confidence_threshold:
            if DEBUG: print("[DEBUG] top score below threshold -> fallback")
            return {"answer": "Tôi không biết - vui lòng liên hệ tổng đài 18001090 hoặc email support@telco.vn", "docs": [d.metadata for d in docs], "hallucinated": False}

        # Build the context: include service_code and page_content for each doc
        context_sections = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            code = d.metadata.get("service_code", "")
            context_sections.append(f"[source: {src} | service_code: {code}]\n{d.page_content}")
        context = "\n\n".join(context_sections)

        # Build prompt and invoke LLM
        prompt = RAG_PROMPT.format(context=context, question=question)
        if DEBUG:
            print("=== Prompt to LLM ===")
            print(prompt[:2000])  # only print head
            print("=== End Prompt ===")
        out = llm.invoke(prompt)
        # ChatGoogleGenerativeAI returns an object; try to extract string
        answer = out.content.strip() if hasattr(out, "content") else str(out).strip()

        # Determine hallucination: ensure at least one [source:] present
        hallucinated = "[source:" not in answer
        return {"answer": answer, "docs": [d.metadata for d in docs], "hallucinated": hallucinated}

    return answer_query

# -------- MAIN runnable example --------
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    # ensure API key set
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: set GOOGLE_API_KEY before running.")
        raise SystemExit(1)

    df = load_service_csv("viettel.csv")
    docs = df_to_documents(df, "viettel.csv")
    chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=50)
    vect = build_vectorstore(chunks)

    qa = make_qa_chain(vect, model_name="gemini-2.0-flash", temperature=0.0, fetch_k=FETCH_K, top_k=TOP_K, confidence_threshold=CONFIDENCE_THRESHOLD)

    # Example queries
    queries = [
        "Để đăng ký gói MXH120 thì soạn tin gửi 191 đúng không?",
        "Nêu cú pháp đăng ký của gói cước có giá 120000 VNĐ và có ưu đãi miễn phí data",
        "Liệt kê tất cả các gói cước có chu kỳ lớn hơn 30 ngày?"
    ]
    for q in queries:
        print("\n---\nQUESTION:", q)
        res = qa(q)
        print("ANSWER:", res["answer"])
        print("HALLUCINATED:", res["hallucinated"])
        print("SOURCES:", res["docs"])
