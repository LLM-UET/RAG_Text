import os
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
from rag_reasoning import (ReasoningDataQueryEngine,ReasoningDataQuery)
# from reasoning import (ReasoningDataQueryEngine,ReasoningDataQuery)
from dotenv import load_dotenv

DEBUG = True
FETCH_K = 20
TOP_K = 5 
CONFIDENCE_THRESHOLD = 1.8

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")
    df["Giá (VNĐ)"] = df["Giá (VNĐ)"].apply(lambda x: int(x) if str(x).strip() != "" else 0)
    df["Chu kỳ (ngày)"] = df["Chu kỳ (ngày)"].apply(lambda x: int(x) if str(x).strip() != "" else 0)
    return df

def df_to_documents(df: pd.DataFrame, source: str) -> List[Document]:
    docs = []
    for _, row in df.iterrows():
        code = str(row.get("Mã dịch vụ", "")).strip()
        text_parts = []
        for col in df.columns:
            val = row[col]
            text_parts.append(f"{col}: {val}")
        text = " . ".join(text_parts)
        docs.append(Document(page_content=text, metadata={"source": source, "service_code": code}))
    return docs



def chunk_documents(docs: List[Document], chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)



def build_vectorstore(docs: List[Document], persist_dir="chroma_db"):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vect = Chroma.from_documents(
        docs,
        embedding=emb,
        ids = [f"{d.metadata.get('service_code','unknown')}_{i}" for i, d in enumerate(docs)],
        persist_directory=persist_dir,
    )
    return vect


def make_reranked_retriever(vectorstore, fetch_k=30, top_k=5):
    # retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    
    def retrieve_with_scores(question: str) -> List[Tuple[Document, float]]:
        # get candidates (list of Document)
        # rq = reasoning_engine.compile(question)
        # if DEBUG:
        #     print(f"[DEBUG] question Compiled: {rq}")

        candidates = retriever.invoke(question)
        if DEBUG:
            print(f"[DEBUG] raw candidates count: {len(candidates)}")
        # dedupe by service_code (keep first occurrence)

        # uniq = []
        # seen = set()
        # for d in candidates:
        #     code = d.metadata.get("service_code")
        #     if code not in seen:
        #         uniq.append(d)
        #         seen.add(code)
        # candidates = uniq
        # if DEBUG:
        #     print(f"[DEBUG] deduped candidates count: {len(candidates)}")
        #     for d in candidates[:10]:
        #         meta = d.metadata
        #         print("[CAND]", meta, "->", d.page_content[:120].replace("\n"," "))
        # if not candidates:
        #     return []

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





RAG_PROMPT = """
🎯 Vai trò:
Bạn là một **trợ lý ảo thông minh** có nhiệm vụ hỗ trợ khách hàng về **các gói cước của nhà mạng Viettel**.

---

🧩 Nhiệm vụ:
Bạn cần **trả lời câu hỏi của người dùng chỉ dựa trên dữ liệu đã cung cấp trong "Ngữ cảnh"**.  
Ngữ cảnh là cơ sở dữ liệu dạng bảng, mỗi hàng tương ứng với một gói cước, các cột mô tả thuộc tính cụ thể.

Các cột của bảng bao gồm:
> Mã dịch vụ, Thời gian thanh toán, Các dịch vụ tiên quyết, Giá (VNĐ), Chu kỳ (ngày),
> 4G tốc độ tiêu chuẩn/ngày, 4G tốc độ cao/ngày, 4G tốc độ tiêu chuẩn/chu kỳ, 4G tốc độ cao/chu kỳ,
> Gọi nội mạng, Gọi ngoại mạng, Tin nhắn, Chi tiết, Tự động gia hạn, Cú pháp đăng ký.

Một số ô có thể trống (tùy gói cước).

---

🧠 Ghi nhớ quy tắc:
1. Nếu dữ liệu "4G tốc độ cao/ngày" hoặc "4G tốc độ tiêu chuẩn/ngày" là số dương  
   → Nghĩa là người dùng được sử dụng tối đa lượng dữ liệu đó mỗi ngày, sau đó reset vào ngày tiếp theo.

2. Nếu **không có dữ liệu theo ngày**, nhưng có dữ liệu theo chu kỳ  
   → Nghĩa là toàn bộ dung lượng đó dùng chung cho toàn chu kỳ.

3. Khi người dùng hỏi về **dung lượng data**, hãy tra cứu các cột sau:
   - "4G tốc độ tiêu chuẩn/ngày"
   - "4G tốc độ cao/ngày"
   - "4G tốc độ tiêu chuẩn/chu kỳ"
   - "4G tốc độ cao/chu kỳ"
   - "Chi tiết"

4. Trong **mọi câu trả lời**, bạn phải trích dẫn tối thiểu các cột:
   - "Mã dịch vụ"
   - "Cú pháp đăng ký"
   - "Giá (VNĐ)"
   - "Chi tiết"

5. Nếu người dùng nói “điện thoại cục gạch”, “nghe gọi ít”, hoặc “ít dùng mạng”  
   → Hiểu là cần **gợi ý gói cước rẻ nhất** (tra cột “Giá (VNĐ)” để chọn giá nhỏ nhất).

6. Nếu người dùng hỏi **gói nào rẻ hơn**,  
   → So sánh cột “Giá (VNĐ)” giữa các gói.

7. Nếu người dùng hỏi **gói nào rẻ nhất**,  
   → Chọn gói có giá trị **MIN của cột “Giá (VNĐ)”**.

8. Nếu có **nhiều bản ghi trùng lặp**,  
   → Chỉ cần tổng hợp và **trả lời tóm tắt nội dung chính một lần**.

---

🚫 Nếu không tìm thấy câu trả lời rõ ràng trong cơ sở dữ liệu, **hãy trả lời chính xác**:
"Tôi không biết - vui lòng liên hệ tổng đài 18001090 hoặc email support@telco.vn"

---

🗣️ Yêu cầu định dạng câu trả lời:
- Viết **ngắn gọn, dễ hiểu cho người dùng phổ thông**.
- Giữ nguyên **ngôn ngữ của người dùng** (ưu tiên tiếng Việt).
- Khi trích dẫn, **luôn ghi rõ nguồn theo dạng [source: filename]**.
- **Tuyệt đối không suy luận hoặc bịa thông tin** không có trong dữ liệu.

---

Ngữ cảnh:
{context}

Câu hỏi người dùng:
{question}

Trả lời:
"""

prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)



def make_qa_chain(vectorstore, model_name:str="gemini-2.0-flash", temperature:float=0.0, fetch_k:int=FETCH_K, top_k:int=TOP_K, confidence_threshold:float=CONFIDENCE_THRESHOLD):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    retriever_with_scores = make_reranked_retriever(vectorstore, fetch_k=fetch_k, top_k=top_k)

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



if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")


    df = load_csv("viettel.csv")
    docs = df_to_documents(df, "viettel.csv")
    chunks = chunk_documents(docs)
    vect = build_vectorstore(chunks)

    qa = make_qa_chain(vect, model_name="gemini-2.0-flash", temperature=0.0, fetch_k=FETCH_K, top_k=TOP_K, confidence_threshold=CONFIDENCE_THRESHOLD)


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