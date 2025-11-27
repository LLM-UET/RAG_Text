import os
import csv
import pandas as pd
import numpy as np
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

def write_local_knowledge_reasoning_query(question: str) -> str | None:
        prompt_temp = """
        Bạn là một trợ lý ảo thông minh của nhà mạng Viettel, có khả năng giải đáp thắc mắc của người dùng.
        Nhiệm vụ: Xác định xem bạn có thể trả lời câu hỏi của người dùng mà chỉ dựa theo kiến thức đã cho hay không, bằng cách truy vấn từ cơ sở dữ liệu.
        Bạn được cung cấp một cơ sở dữ liệu các gói cước (gọi tắt là gói) của Viettel. Đây là cơ sở dữ liệu dạng bảng, mỗi hàng chứa thông tin của một gói, mỗi cột chứa thuộc tính cụ thể của gói đó.
        Một số hàng có thể trống (optional).
        Các cột của bảng bao gồm:
        > Mã dịch vụ, Thời gian thanh toán, Các dịch vụ tiên quyết, Giá (VNĐ), Chu kỳ (ngày),
        > 4G tốc độ tiêu chuẩn/ngày, 4G tốc độ cao/ngày, 4G tốc độ tiêu chuẩn/chu kỳ, 4G tốc độ cao/chu kỳ,
        > Gọi nội mạng, Gọi ngoại mạng, Tin nhắn, Chi tiết, Tự động gia hạn, Cú pháp đăng ký.

        Chú ý 1: Nếu dữ liệu theo ngày là số dương thì nghĩa là một ngày người dùng chỉ được dùng tối đa bấy nhiêu dữ liệu mà thôi, sang ngày khác lại được thêm. Còn nếu không có dữ liệu theo ngày thì nghĩa là người dùng được dùng thoải mái toàn bộ dữ liệu trong chu kỳ mà không bị giới hạn theo ngày, cho đến khi hết dữ liệu trong chu kỳ đó thì phải chờ chu kỳ tiếp theo (nếu gia hạn) mới được tiếp tục sử dụng.
        Chú ý 1b: Nếu người dùng hỏi dung lượng thì cần chọn các cột sau: "4G tốc độ tiêu chuẩn/ngày", "4G tốc độ cao/ngày", "4G tốc độ tiêu chuẩn/chu kỳ", "4G tốc độ cao/chu kỳ", "Chi tiết".
        Chú ý 2: Bạn phải luôn SELECT các cột sau trong mọi trường hợp: "Mã dịch vụ", "Cú pháp", "Giá (VNĐ)", "Chi tiết".
        Chú ý 3: Nếu người dùng nhờ tư vấn cho điện thoại cục gạch, nghe gọi ít hoặc ít sử dụng mạng... thì bạn cần hiểu là phải tìm gói cước rẻ nhất.

        Cú pháp truy cập lấy dữ liệu từ cơ sở dữ liệu như sau:
        SELECT "Tên cột 1", "Tên cột 2"
        WHERE "Tên cột 3" = "Giá trị 3" AND "Tên cột 4" > "Giá trị 4"...
        OR "Tên cột 5" < "Giá trị 5" AND "Tên cột 6" <= "Giá trị 6"...
        OR "Tên cột 7" REACHES MIN
        OR "Tên cột 8" REACHES MAX
        OR "Tên cột 9" CONTAINS "Giá trị 9"...
        ...

        trong đó tên cột và giá trị luôn ở trong dấu nháy (") cho dù đó là giá trị số đi chăng nữa (chẳng hạn "6").
        Tên cột cũng như giá trị sẽ không bao giờ và không được chứa một dấu nháy khác trong đó, nếu không truy vấn sẽ bị coi là sai. Ví dụ "6"" là sai.
        Bạn không cần viết những điều kiện loại bỏ dữ liệu sai ví dụ "4G tốc độ cao/chu kỳ" > 0. Hãy mặc định dữ liệu luôn đúng.
        Bạn không được phép dùng dấu ngoặc đơn như này ( hoặc như này ) để nhóm các biểu thức logic AND-OR. Hãy cố gắng "phá ngoặc" để viết lại câu truy vấn cho dễ hiểu hơn nhé.
        Thứ tự ưu tiên luôn là AND trước rồi mới đến OR.
        Bạn cũng không được phép dùng các toán tử so sánh khác ngoài =, >, <, >=, <=, REACHES MIN, REACHES MAX, CONTAINS.
        Bạn cũng không được phép dùng các toán tử logic khác ngoài AND, OR.
        Bạn cũng không được phép dùng các toán tử khác ngoài SELECT, WHERE.
        Mệnh đề WHERE là bắt buộc.
        Khi người dùng hỏi giá rẻ, giá rẻ nhất thì cần viết query theo kiểu "Giá (VNĐ)" REACHES MIN, chứ không được so sánh với một giá trị cụ thể nào đó, chẳng hạn "Giá (VNĐ)" < 100000.
        Tuy nhiên nếu người dùng hỏi "giá rẻ hơn" thì phải dựa vào lịch sử chat để biết người dùng đang nói tới những gói nào, sau đó xác định gói rẻ hơn trong các gói đó.
        Nếu người dùng hỏi "nhiều data", "data không giới hạn", "miễn phí"... thì nên chọn các gói có "4G tốc độ tiêu chuẩn/ngày" REACHES MIN hoặc "4G tốc độ cao/ngày" REACHES MAX, hoặc cột "Chi tiết" CONTAINS "không giới hạn", "thả ga", "miễn phí" .v.v.

        Trong trường hợp bạn có thể trả lời câu hỏi của người dùng bằng cách tạo một truy vấn dữ liệu như trên, hãy trả về truy vấn.
        Nếu không tạo được truy vấn nhưng câu hỏi vẫn thuộc phạm vi thông tin gói cước, sim thẻ, nhà mạng, giá cả... thì trả về:
            SELECT "Mã dịch vụ", "Cú pháp", "Giá (VNĐ)", "Chi tiết" WHERE "Chi tiết" CONTAINS "<các từ khóa trong câu hỏi của người dùng>"
        Nếu câu hỏi hoàn toàn nằm ngoài phạm vi những thông tin gói cước, sim thẻ... như trên thì trả về IMPOSSIBLE.

        Hãy nghiên cứu các ví dụ dưới đây, và trả lời câu hỏi được đưa ra ở cuối cùng:
        Ví dụ 1:
        - Câu hỏi: Gói cước nào có giá rẻ nhất?
        - Trả lời: SELECT "Mã dịch vụ", "Giá (VNĐ)" WHERE "Giá (VNĐ)" REACHES MIN
        Ví dụ 2:
        - Câu hỏi: Làm thế nào để đăng ký dịch vụ SD70?
        - Trả lời: SELECT "Chi tiết", "Cú pháp", "Mã dịch vụ" WHERE "Mã dịch vụ" = "SD70"
        Ví dụ 3:
        - Câu hỏi: Bạn ơi thế sao thuê bao của tôi cứ tự trừ tiền thế nhỉ, bạn xem giúp tôi số dư còn bao nhiêu với
        - Trả lời: IMPOSSIBLE
        Ví dụ 4:
        - Câu hỏi: Ừ thế xem giúp tôi gói nào để anh lướt mạng thả ga đi, một ngày xem phim đã tốn mấy gigabyte rồi
        - Trả lời: SELECT "Mã dịch vụ", "4G tốc độ tiêu chuẩn/ngày" WHERE "Chi tiết" CONTAINS "lướt mạng thả ga" AND "4G tốc độ tiêu chuẩn/ngày" REACHES MIN
        Ví dụ 5:
        - Câu hỏi: À bạn ơi bên bạn có gói nào rẻ mà lướt mạng thoải mái không, chứ một ngày tôi lướt mạng hết mấy gigabyte rồi
        - Trả lời: SELECT "Mã dịch vụ", "4G tốc độ tiêu chuẩn/ngày" WHERE "Chi tiết" CONTAINS "lướt mạng thoải mái" AND "4G tốc độ tiêu chuẩn/ngày" REACHES MIN AND "Giá (VNĐ)" REACHES MIN

        Câu hỏi: {question}
        """
        # print(f"write_local_knowledge_reasoning_query: prompt: {prompt_temp}")
        print(f"write_local_knowledge_reasoning_query: calling...")
        # prompt_template_query = ChatPromptTemplate.from_template(prompt_temp)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

        prompt = prompt_temp.format(question=question)
        response = llm.invoke(prompt)
        print(f"write_local_knowledge_reasoning_query: response: {response}")
        response_text = response.content.strip()
        # return None if "impossible" in response.content.strip().lower() else response.strip()
        return None if "impossible" in response_text.lower() else response_text

def make_reranked_retriever(df,vectorstore, fetch_k=30, top_k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    reasoning_engine = ReasoningDataQueryEngine(df=df, embedder=emb)

    
    def retrieve_with_scores(question: str) -> List[Tuple[Document, float]]:
        # get candidates (list of Document)
        candidates = retriever.invoke(question)
        if DEBUG:
            print(f"[DEBUG] raw candidates count: {len(candidates)}")

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

        q=write_local_knowledge_reasoning_query(question)
        if DEBUG:
            print("SQL: ", q)
        if q is None or q.strip() == "IMPOSSIBLE":
            return {"type": "docs", "docs": scored_sorted[:top_k]}
    
        try:
            queryObject = reasoning_engine.compile(q)
            table = reasoning_engine.apply(queryObject)
            if DEBUG:
                print("Table:", table)
            return {
                "type": "table",
                "table": table,
                "queryObject": queryObject
            }
        except Exception as e:
            return {"type":"docs", "docs": scored_sorted[:top_k]}
        
        
        # try:
        #     queryObject = reasoning_engine.compile(q)
        #     print("Query Object: ", queryObject)
        # except Exception as e:
        #     print(f"check_local_knowledge_reasoning: query compilation failed: {e}")
        #     print(f"Initial query written by LLM: {q}")
        #     return None

        # try:
        #     queryObject = reasoning_engine.compile(q)
        #     print("Query Object: ", queryObject)
        # except Exception as e:
        #     print(f"check_local_knowledge_reasoning: query compilation failed: {e}")
        #     print(f"Initial query written by LLM: {q}")
        #     return None
        
        # table = reasoning_engine.apply(queryObject)
        # if DEBUG:
        #     print(f"check_local_knowledge_reasoning: applied table: {table}")
        
        # return scored_sorted[:top_k]

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



def make_qa_chain(vectorstore, df, model_name:str="gemini-2.0-flash", temperature:float=0.0, fetch_k:int=FETCH_K, top_k:int=TOP_K, confidence_threshold:float=CONFIDENCE_THRESHOLD):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    # retriever_with_scores = make_reranked_retriever(df, vectorstore, fetch_k=fetch_k, top_k=top_k)
    retriever_handler = make_reranked_retriever(df, vectorstore, fetch_k=fetch_k, top_k=top_k)
    

    def answer_query(question: str):
        result = retriever_handler(question)

        if result["type"] == "table":
            table = result["table"]
            query_obj = result["queryObject"]

            try:
                table_md = table.to_markdown(index=False)
            except:
                table_md = str(table)

            prompt_case1 = f"""
            Bạn là một trợ lý ảo thông minh, là nhân viên chăm sóc khách hàng của Viettel. Bạn có khả năng trả lời câu hỏi của người dùng dựa trên bảng dữ liệu đã cho.
            BẢNG DỮ LIỆU:
            {table_md}
            CÂU HỎI:
            {question}
            """
            if DEBUG:
                print("prompt_case1", prompt_case1)
            out = llm.invoke(prompt_case1)
            if DEBUG:
                print("Output", out)
            answer = out.content.strip()

            return { "answer": answer, "docs": [], "hallucinated": False }
        else: 
            # scored = retriever_with_scores(question)  # List[(Document, score)]
            scored = retriever_handler(question)
            if not scored:
                if DEBUG: print("[DEBUG] No candidates -> fallback")
                return {"answer": "Tôi không biết - vui lòng liên hệ tổng đài 18001090 hoặc email support@telco.vn", "docs": [], "hallucinated": False}
            
            print("DEBUG scored =", scored)
            print("First element =", scored[0], "len =", len(scored[0]))
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
                # table_text = d.metadata.get("filter_table", "")
                # context_sections.append(f"[source: {src} | service_code: {code}]\n{d.page_content}\n{table_text}")
                context_sections.append(f"[source: {src} | service_code: {code}]\n{d.page_content}")
            context = "\n\n".join(context_sections)

            # filtered_table_text = ""
            # for d in docs:
            #     if "filtered_table" in d.metadata:
            #         filtered_table_text = d.metadata["filtered_table"]
            #         break

            # # --- 5. Build full context ---
            # full_context = context
            # if filtered_table_text:
            #     full_context += "\n\n" + filtered_table_text

            # Build prompt and invoke LLM
            prompt = RAG_PROMPT.format(context=context, question=question)
            # if DEBUG:
            #     print("=== Prompt to LLM ===")
            #     print(prompt[:2000])  # only print head
            #     print("=== End Prompt ===")
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

    qa = make_qa_chain(vect, df=df, model_name="gemini-2.0-flash", temperature=0.0, fetch_k=FETCH_K, top_k=TOP_K, confidence_threshold=CONFIDENCE_THRESHOLD)

    

    queries = [
        "Để đăng ký gói MXH120 thì soạn tin gửi 191 đúng không?",
        "Nêu cú pháp đăng ký của gói cước có giá 120000 VNĐ và có ưu đãi miễn phí data",
        "Liệt kê tất cả các gói cước có chu kỳ lớn hơn 30 ngày?",
        "Bạn bao nhiêu tuổi?"
    ]
    for q in queries:
        print("\n---\nQUESTION:", q)
        res = qa(q)
        
        print("ANSWER:", res["answer"])
        print("HALLUCINATED:", res["hallucinated"])
        print("SOURCES:", res["docs"])