import os
from typing import List, Dict
import csv, glob
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder

def load_service_csv(path: str):
    docs = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def get(k): return row.get(k, "").strip()
            code = get("Mã dịch vụ")
            text = (
                f"Gói cước {code} ({get('Thời gian thanh toán')}). "
                f"Giá: {get('Giá (VNĐ)')}đ / {get('Chu kỳ (ngày)')} ngày. "
                f"4G tốc độ cao/ngày: {get('4G tốc độ cao/ngày')}. "
                f"Tự động gia hạn: {get('Tự động gia hạn')}. "
                f"Cú pháp đăng ký: {get('Cú pháp đăng ký')}. "
            )
            if get('Chi tiết'): text += f"Chi tiết: {get('Chi tiết')}. "
            if get('Gọi nội mạng'): text += f"Gọi nội mạng: {get('Gọi nội mạng')}. "
            if get('Gọi ngoại mạng'): text += f"Gọi ngoại mạng: {get('Gọi ngoại mạng')}. "
            docs.append(Document(page_content=text, metadata={"source": path, "service_code": code}))
    return docs

def chunk_documents(docs: List[Document], chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def build_vectorstore(docs: List[Document], persist_dir: str="chroma_db", use_openai=False):
    if use_openai:
        emb = OpenAIEmbeddings()
    else:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vect = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=persist_dir)
    # vect.persist()
    return vect

def make_reranked_retriever(vectorstore, fetch_k=20, top_k=5):
    """
    1. Lấy trước fetch_k candidates từ retriever gốc
    2. Rerank bằng cross-encoder (mạnh hơn cosine similarity)
    3. Trả về top_k tốt nhất
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # tốc độ nhanh, độ chính xác cao

    def retrieve(question: str) -> List[Document]:
        candidates = retriever.invoke(question)
        if not candidates:
            return []
        pairs = [(question, d.page_content) for d in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in ranked[:top_k]]
        return reranked_docs

    return retrieve


RAG_PROMPT = """You are a helpful customer support assistant for a telecom operator.

Answer the user's question using only the provided context.
If the context doesn't contain the answer, reply exactly:
"I don't know - please contact support at 18001090 or support@telco.vn"

Always cite your sources as [source: filename].

Context:
{context}

User question:
{question}

Answer concisely in Vietnamese (or match user's language).
"""

prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)

def make_qa_chain(vectorstore, model_name="gpt-4o-mini", temperature=0.0, use_openai_llm=True):
    if use_openai_llm:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    else:
        raise NotImplementedError("Only OpenAI LLM supported here")

    retriever_fn = make_reranked_retriever(vectorstore)

    rag_chain_lcel = (
        {
            "context": retriever_fn,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    def answer_query(question: str, k:int=5):
        docs = retriever_fn(question)
        # context = "\n\n".join(
        #     [f"[source: {os.path.basename(d.metadata.get('source',''))}]\n{d.page_content}" for d in docs]
        # )
        # prompt = RAG_PROMPT.format(context=context, question=question)
        # out = llm.invoke(prompt)
        # answer = out.content.strip() if hasattr(out, "content") else str(out)

        answer = rag_chain_lcel.invoke(question)

        hallucinated = "[source:" not in answer
        return {
            "answer": answer,
            "docs": docs,
            "hallucinated": hallucinated
        }

    return answer_query

if __name__ == "__main__":
    docs = load_service_csv("viettel.csv")
    # print(docs)
    vect = build_vectorstore(docs)
    # print(vect)
    qa = make_qa_chain(vect)
    # print(qa)

    # q = "Gói MXH120 có phải là gói trả trước không?"
    # q="Gói MXH100 có tự động gia hạn không?"
    # q="Gói V120B có được miễn phí gọi nội mạng không?"
    # q="Gói MXH120 có nội dung gì?"
    # q = "Giá của gói cước MXH100 là bao nhiêu?"
    # q = "Tin nhắn của gói VB90 là gì?"
    # q = "Để đăng ký gói MXH120 thì soạn tin gửi 191 đúng không?"
    q = "Để đăng ký gói MXH100 thì soạn tin gửi 290 đúng không?"
    result = qa(q)

    print("ANSWER:", result["answer"])
    print("HALLUCINATED:", result["hallucinated"])
    print("SOURCES:", [d.metadata for d in result["docs"]])
