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
from langchain_google_genai import ChatGoogleGenerativeAI


from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

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
    # print("-"*10)
    # print("Docs:", docs)
    # print("-"*10)
    return docs

def chunk_documents(docs: List[Document], chunk_size=1024, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def build_vectorstore(docs: List[Document], persist_dir: str="chroma_db", use_openai=False):
    if use_openai:
        emb = OpenAIEmbeddings()
    else:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vect = Chroma.from_documents(documents=docs, embedding=emb, ids=[d.metadata["service_code"] + f"_{i}" for i, d in enumerate(docs)], persist_directory=persist_dir)
    # print(type(vect))
    # vect.persist()
    return vect

def make_reranked_retriever(vectorstore, fetch_k=20, top_k=5):
    """
    1. Lấy trước fetch_k candidates từ retriever gốc
    2. Rerank bằng cross-encoder (mạnh hơn cosine similarity)
    3. Trả về top_k tốt nhất
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # reranker = CrossEncoder("cross-encoder/ms-marco-electra-base", max_length=512)

    def retrieve(question: str) -> List[Document]:
        candidates = retriever.invoke(question)
        # print(candidates)
        unique_candidates = []
        seen = set()
        for d in candidates:
            code = d.metadata.get("service_code")
            if code not in seen:
                unique_candidates.append(d)
                seen.add(code)
        candidates = unique_candidates
        if not candidates:
            return []
        pairs = [(question, d.page_content) for d in candidates]
        # print(pairs)
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        # print(ranked)
        reranked_docs = [doc for doc, _ in ranked[:top_k]]
        # print(reranked_docs)
        # for d in candidates:
        #     print("CAND:", d.metadata, "=>", d.page_content[:80])
        return reranked_docs

    return retrieve


# def make_reranked_retriever(vectorstore, fetch_k=20, top_k=5):
#     """
#     1. Lấy trước fetch_k candidates từ retriever gốc
#     2. Rerank bằng cross-encoder (mạnh hơn cosine similarity)
#     3. Trả về top_k tốt nhất
#     """
#     retriever = vectorstore.as_retriever(search_kwargs={"fetch_k": fetch_k, "k": top_k})
#     reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#     def retrieve(question: str) -> List[Document]:
#         candidates = retriever.invoke(question)
#         if not candidates:
#             return []
#         pairs = [(question, d.page_content) for d in candidates]
#         scores = reranker.predict(pairs)
#         ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
#         reranked_docs = [doc for doc, _ in ranked]
#         return reranked_docs

#     return retrieve


# RAG_PROMPT = """You are a helpful customer support assistant for a telecom operator.

# Answer the user's question using only the provided context.
# If the context doesn't contain the answer, reply exactly:
# "I don't know - please contact support at 18001090 or support@telco.vn"

# Always cite your sources as [source: filename].

# Context:
# {context}

# User question:
# {question}

# Answer concisely in Vietnamese (or match user's language).
# """

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
# print(f"PROMPT_TEMPLATE: {RAG_PROMPT}")

def make_qa_chain(vectorstore, model_name="gemini-2.0-flash", temperature=0.0, use_openai_llm=True):
    if use_openai_llm:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
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

        # Build a plain-text debug prompt from the same template used by the chain
        # and print it so we can inspect what is being passed to the LLM.
        # try:
        #     prompt = RAG_PROMPT.format(context=context, question=question)
        # except Exception:
        #     # Fallback: if formatting fails for any reason, create a minimal prompt
        #     prompt = f"Context:\n{context}\n\nUser question:\n{question}\n"

        # print("=== Prompt sent to LLM ===")
        # print(prompt)
        # print("=== End Prompt ===")

        # Invoke the composed chain (it will construct prompts/messages internally too)
        answer = rag_chain_lcel.invoke(question)

        hallucinated = "[source:" not in answer
        return {
            "answer": answer,
            "docs": docs,
            "hallucinated": hallucinated
        }

    return answer_query

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    docs = load_service_csv("viettel.csv")
    chunks_docs = chunk_documents(docs)
    # print(docs)
    vect = build_vectorstore(chunks_docs)
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
    # q = "Để đăng ký gói MXH100 thì soạn tin gửi 290 đúng không?"
    # q = "Liệt kê các gói cước miễn phí data."
    # q = "Nêu cú pháp đăng ký của gói cước có giá 120.000 VNĐ có ưu đãi miễn phí gọi nội mạng"
    q = "Nêu cú pháp đăng ký của gói cước có giá 120.000 VNĐ và có ưu đãi miễn phí data"
    result = qa(q)

    print("QUESTION", q)
    print("ANSWER:", result["answer"])
    print("HALLUCINATED:", result["hallucinated"])
    print("SOURCES:", [d.metadata for d in result["docs"]])


    # q_array = ["Gói MXH120 có phải là gói trả trước không?", "Gói MXH100 có tự động gia hạn không?", "Gói V120B có được miễn phí gọi nội mạng không?", "Gói MXH120 có nội dung gì?", "Giá của gói cước MXH100 là bao nhiêu?", "Tin nhắn của gói VB90 là gì?", "Để đăng ký gói MXH120 thì soạn tin gửi 191 đúng không?", "Để đăng ký gói MXH100 thì soạn tin gửi 290 đúng không?"]
    
    # q_array = ["Gói cước SD70 có giá bao nhiêu và cung cấp bao nhiêu GB data tốc độ tiêu chuẩn trong một chu kỳ 30 ngày?", "Hãy so sánh gói cước V90B và V120B. Sự khác biệt về giá, data tốc độ cao (tính theo tổng chu kỳ) và phút gọi ngoại mạng là gì?", "Liệt kê tất cả các gói cước có chu kỳ lớn hơn 30 ngày (tức là gói dài hạn) VÀ có ưu đãi miễn phí gọi nội mạng (cụ thể là Miễn phí các cuộc gọi dưới 10 phút hoặc tương đương) HOẶC ưu đãi data tốc độ cao theo ngày là 1GB? Nếu không có gói nào, hãy giải thích tại sao.", "Một người dùng sử dụng gói 12MXH100. Giả sử giá cước không đổi, nếu họ dùng gói cước ngắn hạn MXH100 trong cùng 360 ngày đó, họ sẽ phải trả thêm/bớt bao nhiêu tiền?", "Gói cước nào có ưu đãi đặc biệt là miễn phí thả ga truy cập không giới hạn và những mạng xã hội nào được bao gồm trong ưu đãi này? Nêu cú pháp đăng ký của gói cước có giá 120.000 VNĐ có ưu đãi này.", "Có bao nhiêu gói cước trong bảng không có thông tin chi tiết về cuộc gọi nội mạng (nghĩa là cột Gọi nội mạng bị bỏ trống), và chúng là những gói nào?"]

    # for i, q in enumerate(q_array):
    #     print(f"| Index: {i},| Question: {q} , | Result: {qa(q)}")


    #12h46: Mới thay đổi đoạn prompt và model cross_encoder