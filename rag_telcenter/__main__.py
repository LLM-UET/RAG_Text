import os
import sys
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY not set in environment variables")

import pandas as pd
import numpy as np
from typing import List, Optional #, Dict, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
from rag_reasoning import (ReasoningDataQueryEngine, ReasoningDataQuery)

from .llm import GeminiLLM

DEBUG = True

# default config
FETCH_K = 20
TOP_K = 5 
CONFIDENCE_THRESHOLD = 1.8
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
DOCUMENT_SEPARATOR = "\n---\n"

class RAG:
    """
    Why you use a retriever + reranker combo?

    - Retriever: quickly **narrows down** the search space.

    - Reranker: takes the top-N results (say 10 - 20) and reorders them precisely based on **full query-document interaction**.

    Without a reranker:

    - Embeddings alone may mis-rank very short or very similar documents.

    With a reranker:

    - Even if the embeddings were ambiguous, the cross-encoder can correctly identify the most relevant document.

    => fetch_k is for retriever, top_k is for reranker.
    => fetch_k = N * top_k where N should be >= 2.
    """
    
    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash",
        temperature: float = 0.0,
        fetch_k: int = FETCH_K,
        top_k: int = TOP_K,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        persist_dir: str = "chroma_db",
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        document_separator: str = DOCUMENT_SEPARATOR,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_separator = document_separator
        
        # Initialize components
        self.llm = GeminiLLM(
            model=model_name,
            temperature=temperature,
            # max_tokens=1024,
            api_key=GOOGLE_API_KEY,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.vectorstore: Optional[Chroma] = None
        self.reasoning_engine: Optional[ReasoningDataQueryEngine] = None

    def get_package_fields_interpretation(self) -> str:
        # TODO: API to get this
        return """
        - Mã dịch vụ: Mã định danh duy nhất của gói cước, ví dụ "SD70".
        - Thời gian thanh toán: Thường có hai giá trị "Trả trước" hoặc "Trả sau".
        - Các dịch vụ tiên quyết: Các dịch vụ cần có trước khi đăng ký gói cước này, có thể để trống.
        - Giá (VNĐ): Giá của gói cước trong một chu kỳ, tính theo đồng Việt Nam.
        - Chu kỳ (ngày): Thời gian hiệu lực của gói cước tính theo ngày. Hết chu kỳ sẽ phải gia hạn để tiếp tục sử dụng.
        - 4G tốc độ tiêu chuẩn/ngày: Dung lượng dữ liệu 4G tốc độ tiêu chuẩn mà người dùng nhận được mỗi ngày, được biểu hiện bằng số GB. Nếu sử dụng hết sẽ bị giảm tốc độ.
        - 4G tốc độ cao/ngày
        - 4G tốc độ tiêu chuẩn/chu kỳ
        - 4G tốc độ cao/chu kỳ
        - Gọi nội mạng: Chi tiết ưu đãi gọi nội mạng trong chu kỳ, ví dụ "Miễn phí 30 phút gọi"
        - Gọi ngoại mạng
        - Tin nhắn: Chi tiết ưu đãi tin nhắn trong chu kỳ.
        - Chi tiết: Mô tả thêm về gói cước, bao gồm các ưu đãi, điều kiện sử dụng, giới hạn...
        - Tự động gia hạn: Cho biết gói cước có tự động gia hạn sau khi hết chu kỳ hay không. Nhận giá trị "Có" hoặc "Không".
        - Cú pháp đăng ký: Hướng dẫn cú pháp SMS hoặc thao tác để đăng ký gói cước.
        """
        
    def update_dataframe(self, df: pd.DataFrame, source: str):
        self.df = df.copy()
        self.df.columns = [c.strip() for c in self.df.columns]
        self.df = self.df.fillna("")
        
        if "Giá (VNĐ)" in self.df.columns:
            self.df["Giá (VNĐ)"] = self.df["Giá (VNĐ)"].apply(
                lambda x: int(x) if str(x).strip() != "" else 0
            )
        if "Chu kỳ (ngày)" in self.df.columns:
            self.df["Chu kỳ (ngày)"] = self.df["Chu kỳ (ngày)"].apply(
                lambda x: int(x) if str(x).strip() != "" else 0
            )
        
        docs = self._df_to_documents(self.df, source)
        chunks = self._chunk_documents(docs)
        self.vectorstore = self._build_vectorstore(chunks)
        
        self.reasoning_engine = ReasoningDataQueryEngine(
            df=self.df,
            embedder=lambda x: np.array(self.embeddings.embed_query(x)),
        )
        
        if DEBUG:
            print(f"[RAG] Updated dataframe with {len(self.df)} rows, {len(docs)} docs, {len(chunks)} chunks")
    
    def _df_to_documents(self, df: pd.DataFrame, source: str) -> List[Document]:
        docs = []
        for _, row in df.iterrows():
            code = str(row.get("Mã dịch vụ", "")).strip()
            text_parts = []
            for col in df.columns:
                val = row[col]
                text_parts.append(f"{col}: {val}")
            text = " . ".join(text_parts)
            docs.append(Document(
                page_content=text,
                metadata={"source": source, "service_code": code}
            ))
        return docs
    
    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[self.document_separator],
        )
        return splitter.split_documents(docs)
    
    def _build_vectorstore(self, docs: List[Document]) -> Chroma:
        vect = Chroma.from_documents(
            docs,
            embedding=self.embeddings,
            ids=[d.metadata.get('service_code', f'unknown_{i}') for i, d in enumerate(docs)],
            persist_directory=self.persist_dir,
        )
        return vect
    
    def query_vectordb(self, query: str) -> str:
        """
        Trả về context gần nhất với query.
        Bạn cần tự nhúng thông tin chat history liên quan vào query nếu cần.
        """
        if self.vectorstore is None:
            raise Exception("Vectorstore chưa được khởi tạo. Gọi update_dataframe() trước.")
        
        # Use similarity_search_with_score to get top fetch_k candidates with their scores
        candidates_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k)
        
        if DEBUG:
            print(f"[query_vectordb] [retriever] candidates count: {len(candidates_with_scores)}")
            print("[query_vectordb] [retriever] candidates taken and scores:")
            for doc, sc in candidates_with_scores:
                print(f"  {doc.metadata.get('service_code')} -> {sc:.4f}")
        
        if not candidates_with_scores:
            raise Exception("Không tìm thấy candidates từ vectordb")
        
        # Rerank using cross-encoder
        candidates = [doc for doc, _ in candidates_with_scores]
        pairs = [(query, d.page_content) for d in candidates]
        rerank_scores = self.reranker.predict(pairs)
        scored = list(zip(candidates, [float(s) for s in rerank_scores]))
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        
        # Take only top_k after reranking
        scored_sorted = scored_sorted[:self.top_k]
        
        if DEBUG:
            print("[query_vectordb] [reranker] top taken and scored:")
            for doc, sc in scored_sorted:
                print(f"  {doc.metadata.get('service_code')} -> {sc:.4f}")
        
        # Check confidence threshold
        top_score = float(scored_sorted[0][1]) if scored_sorted else 0.0
        if top_score < self.confidence_threshold:
            # TODO: What is this?
            raise Exception(f"Top score {top_score:.4f} below threshold {self.confidence_threshold}")
        
        # Build context from top_k documents (already sliced to top_k above)
        top_docs = [doc for doc, _ in scored_sorted]
        context_sections = []
        for d in top_docs:
            src = d.metadata.get("source", "unknown")
            code = d.metadata.get("service_code", "")
            context_sections.append(f"[source: {src} | service_code: {code}]\n{d.page_content}")
        
        context = "\n---\n".join(context_sections)
        return context
    
    def query_reasoning(self, chat_history: str, query: str) -> str:
        if self.reasoning_engine is None:
            raise Exception("Reasoning engine chưa được khởi tạo. Gọi update_dataframe() trước.")
        
        # Generate SQL-like query using LLM
        sql_query = self._get_reasoning_query(chat_history, query)
        
        if DEBUG:
            print(f"[query_reasoning] Generated SQL: {sql_query}")
        
        if sql_query is None:
            raise Exception("LLM không thể sinh SQL query cho câu hỏi này")
        
        # Compile and execute query
        try:
            query_object = self.reasoning_engine.compile(sql_query)
            result_table = self.reasoning_engine.apply(query_object)
            
            if DEBUG:
                print(f"[query_reasoning] Result table shape: {result_table}")
            
            if not result_table:
                raise Exception("Query execution trả về bảng rỗng")
            
            # Convert table to markdown
            try:
                table_md = result_table.to_df().to_markdown(index=False)
            except:
                table_md = str(result_table)
            
            return table_md
            
        except Exception as e:
            raise Exception(f"Không thể compile hoặc execute query: {e}")
    
    def _get_reasoning_query(self, chat_history: str, question: str) -> Optional[str]:
        """
        Sử dụng LLM để sinh pseudo-SQL query từ câu hỏi tự nhiên.
        
        Returns:
            SQL query string hoặc None nếu IMPOSSIBLE
        """
        prompt_temp = f"""
        Bạn là một trợ lý ảo thông minh của nhà mạng Viettel, có khả năng giải đáp thắc mắc của người dùng.
        Nhiệm vụ: Xác định xem bạn có thể trả lời câu hỏi của người dùng mà chỉ dựa theo kiến thức đã cho hay không, bằng cách truy vấn từ cơ sở dữ liệu.
        Bạn được cung cấp một cơ sở dữ liệu các gói cước (gọi tắt là gói) của Viettel. Đây là cơ sở dữ liệu dạng bảng, mỗi hàng chứa thông tin của một gói, mỗi cột chứa thuộc tính cụ thể của gói đó.
        Một số hàng có thể trống (optional).
        Các cột của bảng bao gồm:

        {self.get_package_fields_interpretation()}

        Chú ý 1: Nếu người dùng hỏi dung lượng thì cần chọn các cột sau: "4G tốc độ tiêu chuẩn/ngày", "4G tốc độ cao/ngày", "4G tốc độ tiêu chuẩn/chu kỳ", "4G tốc độ cao/chu kỳ", "Chi tiết".
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

        Hãy nghiên cứu các ví dụ về truy vấn và câu trả lời dưới đây, từ đó trả lời truy vấn đầu vào được đưa ra ở cuối.

        Ví dụ 1:
        - Lịch sử chat: Không có
        - Câu hỏi: Gói cước nào có giá rẻ nhất?
        - Trả lời: SELECT "Mã dịch vụ", "Giá (VNĐ)" WHERE "Giá (VNĐ)" REACHES MIN
        Ví dụ 2:
        - Lịch sử chat: Người dùng hỏi giá gói cước rẻ nhất, trợ lý ảo trả lời gói SD70 (70.000đ/tháng, 30GB) là rẻ nhất.
        - Câu hỏi: Làm thế nào để đăng ký gói đấy?
        - Trả lời: SELECT "Chi tiết", "Cú pháp" và "Mã dịch vụ" WHERE "Mã dịch vụ" = "SD70"
        Ví dụ 3:
        - Lịch sử chat: Người dùng hỏi về cách kiểm tra số dư tài khoản, trợ lý ảo trả lời cần bấm *101# để kiểm tra.
        - Câu hỏi: Em ơi thế sao thuê bao của anh cứ tự trừ tiền thế nhỉ, em xem giúp anh số dư còn bao nhiêu với
        - Trả lời: IMPOSSIBLE
        Ví dụ 4:
        - Lịch sử chat: Không có
        - Câu hỏi: Ừ thế xem giúp anh gói nào để anh lướt mạng thả ga đi, một ngày xem phim đã tốn mấy gigabyte rồi
        - Trả lời: SELECT "Mã dịch vụ", "4G tốc độ tiêu chuẩn/ngày" WHERE "Chi tiết" CONTAINS "lướt mạng thả ga" AND "4G tốc độ tiêu chuẩn/ngày" REACHES MIN
        Ví dụ 5:
        - Lịch sử chat: Không có
        - Câu hỏi: À em ơi bên em có gói nào rẻ mà lướt mạng thoải mái không, chứ một ngày anh lướt mạng hết mấy gigabyte rồi
        - Trả lời: SELECT "Mã dịch vụ", "4G tốc độ tiêu chuẩn/ngày" WHERE "Chi tiết" CONTAINS "lướt mạng thoải mái" AND "4G tốc độ tiêu chuẩn/ngày" REACHES MIN AND "Giá (VNĐ)" REACHES MIN
        Ví dụ 6:
        - Lịch sử chat:
            Trợ lý ảo gợi ý gói 6MXH100 (180GB/tháng) và 12MXH100 (360GB/tháng) vì phù hợp nhu cầu xem phim nhiều.
            Người dùng hỏi cụ thể về ưu đãi của các gói 6MXH100 và 12MXH100, muốn tìm gói rẻ mà nhiều data.
            Trợ lý ảo gợi ý thêm các gói SD70 (70.000đ/tháng, 30GB), V90B (90.000đ/tháng, 30GB) và MXH100 (100.000đ/tháng, 30GB), lưu ý data có thể không đủ nếu xem phim nhiều.
            Người dùng muốn được tư vấn gói 70.000đ/tháng.
            Trợ lý ảo xác nhận gói SD70 (70.000đ/tháng, 30GB) phù hợp yêu cầu, nhưng lưu ý 30GB có thể không đủ cho nhu cầu xem phim nhiều. Gợi ý tham khảo các gói data lớn hơn nếu cần.
        - Câu hỏi: À vậy gói này đăng ký thế nào em nhỉ?
        - Trả lời: SELECT "Mã dịch vụ", "Cú pháp", "Giá (VNĐ)", "Chi tiết" WHERE "Mã dịch vụ" = "SD70"

        Hãy trả lời truy vấn dưới đây:
        - Lịch sử chat: {chat_history}
        
        - Câu hỏi: {question}
        """
        
        if DEBUG:
            print(f"[_get_reasoning_query] calling LLM...")
        
        prompt = prompt_temp.format(question=question)
        response = self.llm.call(prompt)
        
        if DEBUG:
            print(f"[_get_reasoning_query] response: {response[:200]}...")
        
        response_text = response.strip()
        return None if "impossible" in response_text.lower() else response_text


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")
    if "Giá (VNĐ)" in df.columns:
        df["Giá (VNĐ)"] = df["Giá (VNĐ)"].apply(lambda x: int(x) if str(x).strip() != "" else 0)
    if "Chu kỳ (ngày)" in df.columns:
        df["Chu kỳ (ngày)"] = df["Chu kỳ (ngày)"].apply(lambda x: int(x) if str(x).strip() != "" else 0)
    return df


if __name__ == "__main__":
    # Initialize RAG system
    rag = RAG()
    
    # Load data and update RAG
    df = load_csv("viettel.csv")
    rag.update_dataframe(df, source="viettel.csv")
    
    # Test queries
    queries = [
        "Để đăng ký gói MXH120 thì soạn tin gửi 191 đúng không?",
        "Nêu cú pháp đăng ký của gói cước có giá 120000 VNĐ và có ưu đãi miễn phí data",
        "Liệt kê tất cả các gói cước có chu kỳ lớn hơn 30 ngày?",
        "Bạn bao nhiêu tuổi?"
    ]
    
    for q in queries:
        print("\n" + "="*80)
        print(f"QUESTION: {q}")
        print("-"*80)
        
        # vectordb
        try:
            context = rag.query_vectordb(query=q) # TODO: include chat history
            print(f"[VECTORDB] Success!")
            print(f"Context preview:\n{context[:500]}...")
        except Exception as e2:
            print(f"[VECTORDB] Failed: {e2}")

            # reasoning
            try:
                context = rag.query_reasoning(chat_history="", query=q)
                print(f"[REASONING] Success!")
                print(f"Result table:\n{context[:500]}...")
            except Exception as e:
                print(f"[REASONING] Failed: {e}")

                # TODO: Who is Partner?
                print("Answer: Xin lỗi, tôi không thể trả lời câu hỏi này. Bạn có muốn tôi chuyển tiếp tới tư vấn viên tổng đài của partner để được hỗ trợ không?")
