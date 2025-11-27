import os
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_reasoning import (ReasoningDataQueryEngine, ReasoningDataQuery)
from dotenv import load_dotenv

DEBUG = True
FETCH_K = 20
TOP_K = 5 
CONFIDENCE_THRESHOLD = 1.8
PATH = "viettel.csv"


class RAG:
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        fetch_k: int = FETCH_K,
        top_k: int = TOP_K,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        persist_dir: str = "chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.vectorstore: Optional[Chroma] = None
        self.reasoning_engine: Optional[ReasoningDataQueryEngine] = None
        
    def update_dataframe(self, df: pd.DataFrame):
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
        
        docs = self._df_to_documents(self.df, PATH)
        chunks = self._chunk_documents(docs)
        self.vectorstore = self._build_vectorstore(chunks)
        
        self.reasoning_engine = ReasoningDataQueryEngine(df=self.df, embedder=self.embeddings.embed_query)
        
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
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(docs)
    
    def _build_vectorstore(self, docs: List[Document]) -> Chroma:
        vect = Chroma.from_documents(
            docs,
            embedding=self.embeddings,
            ids=[f"{d.metadata.get('service_code','unknown')}_{i}" for i, d in enumerate(docs)],
            persist_directory=self.persist_dir,
        )
        return vect
    
    def query_vectordb(self, chat_history: str, query: str) -> str:
        if self.vectorstore is None:
            raise Exception("Vectorstore chưa được khởi tạo. Gọi update_dataframe() trước.")
        
        # Use similarity_search_with_score to get top fetch_k candidates with their scores
        candidates_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k)
        
        if DEBUG:
            print(f"[query_vectordb] raw candidates count: {len(candidates_with_scores)}")
        
        if not candidates_with_scores:
            raise Exception("Không tìm thấy candidates từ vectordb")
        
        # Deduplicate by service_code (keep first occurrence)
        unique_docs_with_scores = []
        seen_codes = set()
        for doc, vec_score in candidates_with_scores:
            code = doc.metadata.get("service_code")
            if code not in seen_codes:
                unique_docs_with_scores.append((doc, vec_score))
                seen_codes.add(code)
        
        if DEBUG:
            print(f"[query_vectordb] deduped candidates count: {len(unique_docs_with_scores)}")
        
        # Rerank using cross-encoder
        candidates = [doc for doc, _ in unique_docs_with_scores]
        pairs = [(query, d.page_content) for d in candidates]
        rerank_scores = self.reranker.predict(pairs)
        scored = list(zip(candidates, [float(s) for s in rerank_scores]))
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        
        # Take only top_k after reranking
        scored_sorted = scored_sorted[:self.top_k]
        
        if DEBUG:
            print("[query_vectordb] top scored:")
            for doc, sc in scored_sorted:
                print(f"  {doc.metadata.get('service_code')} -> {sc:.4f}")
        
        # Check confidence threshold
        top_score = float(scored_sorted[0][1]) if scored_sorted else 0.0
        if top_score < self.confidence_threshold:
            raise Exception(f"Top score {top_score:.4f} below threshold {self.confidence_threshold}")
        
        # Build context from top_k documents (already sliced to top_k above)
        top_docs = [doc for doc, _ in scored_sorted]
        context_sections = []
        for d in top_docs:
            src = d.metadata.get("source", "unknown")
            code = d.metadata.get("service_code", "")
            context_sections.append(f"[source: {src} | service_code: {code}]\n{d.page_content}")
        
        context = "\n\n".join(context_sections)
        return context
    
    def query_reasoning(self, chat_history: str, query: str) -> str:
        if self.reasoning_engine is None:
            raise Exception("Reasoning engine chưa được khởi tạo. Gọi update_dataframe() trước.")
        
        # Generate SQL-like query using LLM
        sql_query = self._write_local_knowledge_reasoning_query(query)
        
        if DEBUG:
            print(f"[query_reasoning] Generated SQL: {sql_query}")
        
        if sql_query is None or sql_query.strip().upper() == "IMPOSSIBLE":
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
                table_md = result_table.to_markdown(index=False)
            except:
                table_md = str(result_table)
            
            return table_md
            
        except Exception as e:
            raise Exception(f"Không thể compile hoặc execute query: {e}")
    
    def _write_local_knowledge_reasoning_query(self, question: str) -> Optional[str]:
        """
        Sử dụng LLM để sinh pseudo-SQL query từ câu hỏi tự nhiên.
        
        Returns:
            SQL query string hoặc None nếu IMPOSSIBLE
        """
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
        
        if DEBUG:
            print(f"[_write_local_knowledge_reasoning_query] calling LLM...")
        
        prompt = prompt_temp.format(question=question)
        response = self.llm.invoke(prompt)
        
        if DEBUG:
            print(f"[_write_local_knowledge_reasoning_query] response: {response.content[:200]}...")
        
        response_text = response.content.strip()
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
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    # Initialize RAG system
    rag = RAG(
        model_name="gemini-2.0-flash",
        temperature=0.0,
        fetch_k=FETCH_K,
        top_k=TOP_K,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Load data and update RAG
    df = load_csv("viettel.csv")
    rag.update_dataframe(df)
    
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
        
        # Test vectordb only
        try:
            context = rag.query_vectordb(chat_history="", query=q)
            print(f"[VECTORDB] Success!")
            print(f"Context:\n{context}\n")
        except Exception as e:
            print(f"[VECTORDB] Failed: {e}")
            print("Answer: Tôi không biết - vui lòng liên hệ tổng đài 18001090")
