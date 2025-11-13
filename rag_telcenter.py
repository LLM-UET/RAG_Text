import os
from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma


import json, csv, glob, pathlib

def load_faq_csv(path: str) -> List[Document]:
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            content = f"Q: {r.get('question','')}\nA: {r.get('answer','')}"
            docs.append(Document(page_content=content, metadata={"source": path, "type":"faq"}))
    return docs

def load_plain_text_file(path: str) -> Document:
    text = open(path, encoding="utf-8").read()
    return Document(page_content=text, metadata={"source": path, "type":"doc"})

def load_kb(dir_path: str) -> List[Document]:
    docs=[]
    for p in glob.glob(os.path.join(dir_path, "**/*"), recursive=True):
        if p.endswith(".csv"):
            docs.extend(load_faq_csv(p))
        elif p.endswith(".txt") or p.endswith(".md"):
            docs.append(load_plain_text_file(p))
    return docs

def load_service_csv(path: str):
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Tạo mô tả thân thiện
            text = (
                f"Gói cước {row['Mã dịch vụ']} ({row['Thời gian thanh toán']}). "
                f"Giá: {row['Giá (VNĐ)']}đ / {row['Chu kỳ (ngày)']} ngày. "
                f"4G tốc độ cao/ngày: {row['4G tốc độ cao/ngày']}. "
                f"Tự động gia hạn: {row['Tự động gia hạn']}. "
                f"Cú pháp đăng ký: {row['Cú pháp đăng ký']}. "
            )
            # Thêm chi tiết nếu có
            if row['Chi tiết']:
                text += f"Chi tiết: {row['Chi tiết']}. "
            if row['Gọi nội mạng']:
                text += f"Gọi nội mạng: {row['Gọi nội mạng']}. "
            if row['Gọi ngoại mạng']:
                text += f"Gọi ngoại mạng: {row['Gọi ngoại mạng']}. "

            doc = Document(page_content=text, metadata={"source": path, "service_code": row['Mã dịch vụ']})
            docs.append(doc)
    return docs

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs: List[Document], chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def build_vectorstore(docs: List[Document], persist_dir: str="chroma_db", use_openai=False):
    if use_openai:
        emb = OpenAIEmbeddings()
    else:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=persist_dir)
    try:
        vect.persist()
    except Exception:
        pass
    return vect

def make_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(search_kwargs={"k":k})

RAG_PROMPT = """You are a helpful customer support assistant for a telecom operator.
Answer the user's question using only the provided context. If the context doesn't contain the answer,
reply "I don't know - please contact support" and provide next steps (phone/email/URL).

Context:
{context}

User question:
{question}

Answer concisely in Vietnamese (or match user's language). Cite the source(s) using metadata like [source: filename] when relevant.
"""

prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)

def make_qa_chain(vectorstore, model_name="gpt-4o-mini", temperature=0.0, use_openai_llm=True):
    if use_openai_llm:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    else:
        raise NotImplementedError("Plug a local LLM wrapper here (e.g. transformers)")

    retriever = make_retriever(vectorstore)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        |StrOutputParser()
    )
    def answer_query(question: str, k:int=5):
        docs = retriever.get_relevant_documents(question) if hasattr(retriever, "get_relevant_documents") else retriever.invoke(question)
        
        out = rag_chain.invoke(question)
        return out, docs
    return answer_query

if __name__ == "__main__":
    # KB_DIR = "./kb"
    # raw_docs = load_kb(KB_DIR)
    # chunks = chunk_documents(raw_docs)
    # vect = build_vectorstore(chunks, persist_dir="./chroma_persist", use_openai=False)

    docs = load_service_csv("viettel.csv")
    vect = build_vectorstore(docs)
    qa = make_qa_chain(vect, use_openai_llm=True)
    # q="Gói MXH100 có tự động gia hạn không?"
    q="Gói MXH120 có phải gói cước trả trước không?"
    ans, sources = qa(q)
    print("ANSWER:", ans)
    print("SOURCES:", [d.metadata for d in sources])
