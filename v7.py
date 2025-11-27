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
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
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
    return docs

def chunk_documents(docs: List[Document], chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

class HybridEmbeddings:
    def __init__(self, semantic_model_name="sentence-transformers/all-mpnet-base-v2"):
        self.semantic_model = HuggingFaceEmbeddings(model_name=semantic_model_name)
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))

        self.vectorizer.fit(["dummy"])

    def fit_lexical(self, texts: List[str]):
        """Huấn luyện vectorizer TF-IDF với toàn bộ corpus để dùng cho lexical embedding"""
        self.vectorizer.fit(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sem_emb = self.semantic_model.embed_documents(texts)
        lex_emb = self.vectorizer.transform(texts).toarray()
        if lex_emb.shape[1] < len(sem_emb[0]):
            pad_width = len(sem_emb[0]) - lex_emb.shape[1]
            lex_emb = np.pad(lex_emb, ((0, 0), (0, pad_width)))
        elif lex_emb.shape[1] > len(sem_emb[0]):
            lex_emb = lex_emb[:, :len(sem_emb[0])]

        return (np.array(sem_emb) + lex_emb).tolist()

    def embed_query(self, text: str) -> List[float]:
        sem_vec = self.semantic_model.embed_query(text)
        lex_vec = self.vectorizer.transform([text]).toarray()[0]
        if len(lex_vec) < len(sem_vec):
            lex_vec = np.pad(lex_vec, (0, len(sem_vec) - len(lex_vec)))
        elif len(lex_vec) > len(sem_vec):
            lex_vec = lex_vec[:len(sem_vec)]
        return (np.array(sem_vec) + lex_vec).tolist()

def build_vectorstore(docs: List[Document], persist_dir: str="chroma_db", use_openai=False, hybrid=True):
    if use_openai:
        emb = OpenAIEmbeddings()
    else:
        if hybrid:
            print(">>> Using Hybrid Embeddings (TF-IDF + all-mpnet-base-v2)")
            emb = HybridEmbeddings(semantic_model_name="sentence-transformers/all-mpnet-base-v2")            
            texts = [d.page_content for d in docs]
            emb.fit_lexical(texts)
        else:
            emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
 
    vect = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=persist_dir)
    # vect.persist()
    return vect

def extract_service_code(q: str):
    m = re.search(r"\b[A-Z]{1,5}\d+[A-Z]?\b", q.upper())
    return m.group(0) if m else None

def extract_registration_number(q: str):
    m = re.search(r"g[ií]i?\s*(?:mã\s*)?(?:số\s*)?(\d{2,4})", q.lower())
    if m:
        return m.group(1)
    m2 = re.search(r"gửi\s+(\d{2,4})", q.lower())
    return m2.group(1) if m2 else None

def deduplicate_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    uniq = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("service_code"), d.page_content.strip())
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq

def make_reranked_retriever(vectorstore, fetch_k=50, top_k=5):
    """
    1. Lấy trước fetch_k candidates từ retriever gốc
    2. Rerank bằng cross-encoder (mạnh hơn cosine similarity)
    3. Trả về top_k tốt nhất
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(question: str) -> List[Document]:
        code = extract_service_code(question)
        reg_num = extract_registration_number(question)

        candidates = retriever.invoke(question)
        if not candidates:
            return []
        
        candidates = deduplicate_docs(candidates)
        
        if code:
            exact_docs = [d for d in candidates if d.metadata.get("service_code") == code]
            if exact_docs: 
                candidates = exact_docs
        pairs = [(question, d.page_content) for d in candidates]
        scores = reranker.predict(pairs)

        boosted = []
        for d, s in zip(candidates, scores):
            boost = 0.0
            pc = d.page_content.upper()
            if code and code in pc:
                boost += 2.0
            if reg_num and reg_num in d.page_content:
                boost += 1.5

            boosted.append((d,s+boost))

        ranked = sorted(boosted, key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in ranked[:top_k]]

        seen_codes = set()
        final = []
        for d in reranked_docs:
            sc = d.metadata.get("service_code")
            if sc not in seen_codes:
                final.append(d)
                seen_codes.add(sc)

        return final

    return retrieve


# RAG_PROMPT = """
# Vai trò: Bạn là một trợ lý ảo thông minh có khả năng giải đáp thắc mắc của khách hàng cho nhà mạng viễn thông.

# Nhiệm vụ: Xác định xem bạn có thể trả lời câu hỏi của người dùng mà chỉ dựa theo kiến thức đã cho hay không, bằng cách truy vấn từ cơ sở dữ liệu.
# Bạn được cung cấp một cơ sở dữ liệu các gói cước của Viettel. Đây là cơ sở dữ liệu dạng bảng, mỗi hàng chứa thông tin của một gói, mỗi cột chứa thuộc tính cụ thể của gói đó.
# Một số hàng có thể trống (optional).
# Các cột của bảng bao gồm: Mã dịch vụ,Thời gian thanh toán,Các dịch vụ tiên quyết,Giá (VNĐ),Chu kỳ (ngày),4G tốc độ tiêu chuẩn/ngày,4G tốc độ cao/ngày,4G tốc độ tiêu chuẩn/chu kỳ,4G tốc độ cao/chu kỳ,Gọi nội mạng,Gọi ngoại mạng,Tin nhắn,Chi tiết,Tự động gia hạn,Cú pháp đăng ký

# Chú ý 1: Nếu dữ liệu theo ngày là số dương thì nghĩa là một ngày người dùng chỉ được dùng tối đa bấy nhiêu dữ liệu mà thôi, sang ngày khác lại được thêm. Còn nếu không có dữ liệu theo ngày thì nghĩa là người dùng được dùng thoải mái toàn bộ dữ liệu trong chu kỳ mà không bị giới hạn theo ngày, cho đến khi hết dữ liệu trong chu kỳ đó thì phải chờ chu kỳ tiếp theo (nếu gia hạn) mới được tiếp tục sử dụng.
# Chú ý 1b: Nếu người dùng hỏi dung lượng thì cần chọn các cột sau: "4G tốc độ tiêu chuẩn/ngày", "4G tốc độ cao/ngày", "4G tốc độ tiêu chuẩn/chu kỳ", "4G tốc độ cao/chu kỳ", "Chi tiết".
# Chú ý 2: Bạn phải luôn truy vấn các cột sau trong mọi trường hợp: "Mã dịch vụ", "Cú pháp", "Giá (VNĐ)" và "Chi tiết".
# Chú ý 3: Nếu người dùng nhờ tư vấn cho điện thoại cục gạch, nghe gọi ít hoặc ít sử dụng mạng... thì bạn cần hiểu là phải tìm gói cước rẻ nhất.
# Chú ý 4: Nếu người dùng hỏi gói nào rẻ hơn, thì chỉ cần so sánh cột "Giá (VNĐ)
# Chú ý 5: Nếu người dùng hỏi gói nào rẻ nhất, thì cần lấy giá trị MIN của cột "Giá (VNĐ)"

# Nếu không tìm thấy câu trả lời rõ ràng trong cơ sở dữ liệu, hãy trả lời chính xác:
# "Tôi không biết - vui lòng liên hệ tổng đài 18001090 hoặc email support@telco.vn"


# Hãy trả lời:
# 1. Ngắn gọn, rõ ràng, dễ hiểu cho người dùng phổ thông.
# 2. Giữ nguyên ngôn ngữ của người dùng (ưu tiên tiếng Việt).
# 3. Khi trích dẫn thông tin, **luôn ghi rõ nguồn theo dạng [source: filename]**.
# 4. Không suy luận hoặc bịa thông tin không có trong cơ sở dữ liệu.
# 5. Nếu tìm thấy nhiều bản ghi trùng nhau, chỉ cần tóm gọn nội dung chính 1 lần.

# ---

# Ngữ cảnh:
# {context}

# Câu hỏi người dùng:
# {question}

# Trả lời:
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

def make_qa_chain(vectorstore, model_name="gemini-2.0-flash", temperature=0.0, use_openai_llm=True):
    if use_openai_llm:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        raise NotImplementedError("Only OpenAI LLM supported here")

    retriever_fn = make_reranked_retriever(vectorstore)

    def answer_query(question: str, k:int=5):
        docs = retriever_fn(question)

        context_pieces = []
        for d in docs:
            text = d.page_content.strip()
            context_pieces.append(f"[source: {os.path.basename(d.metadata.get('source',''))}] {text}")

        context = "\n\n".join(context_pieces) if context_pieces else ""
        prompt = RAG_PROMPT.format(context=context, question=question)
        resp = llm.invoke(prompt)
        answer = resp.content.strip() if hasattr(resp, "content") else str(resp)

        code = extract_service_code(question)
        reg_num = extract_registration_number(question)
        hallucinated = "[source:" not in answer or (code and code not in answer and not any(code in d.page_content for d in docs))
        return {"answer": answer, "docs": docs, "hallucinated": hallucinated}

    return answer_query


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    docs = load_service_csv("viettel.csv")
    # print(docs)
    vect = build_vectorstore(docs, hybrid=True)
    print(f"VECTOR: {vect}")
    qa = make_qa_chain(vect)
    print(f"QA: {qa}")

    q_array = ["Gói cước SD70 có giá bao nhiêu và cung cấp bao nhiêu GB data tốc độ tiêu chuẩn trong một chu kỳ 30 ngày?", "Hãy so sánh gói cước V90B và V120B. Sự khác biệt về giá, data tốc độ cao (tính theo tổng chu kỳ) và phút gọi ngoại mạng là gì?", "Liệt kê tất cả các gói cước có chu kỳ lớn hơn 30 ngày (tức là gói dài hạn) VÀ có ưu đãi miễn phí gọi nội mạng (cụ thể là Miễn phí các cuộc gọi dưới 10 phút hoặc tương đương) HOẶC ưu đãi data tốc độ cao theo ngày là 1GB? Nếu không có gói nào, hãy giải thích tại sao.", "Một người dùng sử dụng gói 12MXH100. Giả sử giá cước không đổi, nếu họ dùng gói cước ngắn hạn MXH100 trong cùng 360 ngày đó, họ sẽ phải trả thêm/bớt bao nhiêu tiền?", "Gói cước nào có ưu đãi đặc biệt là miễn phí thả ga truy cập không giới hạn và những mạng xã hội nào được bao gồm trong ưu đãi này? Nêu cú pháp đăng ký của gói cước có giá 120.000 VNĐ có ưu đãi này.", "Có bao nhiêu gói cước trong bảng không có thông tin chi tiết về cuộc gọi nội mạng (nghĩa là cột Gọi nội mạng bị bỏ trống), và chúng là những gói nào?"]
    
    for i, q in enumerate(q_array):
        result = qa(q)
        print(f"| Index: {i},| Question: {q} , | Result: {result}")
        print("HALLUCINATED:", result["hallucinated"])
        print("SOURCES:", [d.metadata for d in result["docs"]])

    # q = "Để đăng ký gói MXH120 thì soạn tin gửi 191 đúng không?"
    # result = qa(q)

    # print("ANSWER:", result["answer"])
    # print("HALLUCINATED:", result["hallucinated"])
    # print("SOURCES:", [d.metadata for d in result["docs"]])