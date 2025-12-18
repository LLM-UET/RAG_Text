import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY not set in environment variables")


from .rag_reasoning import ReasoningDataQuery, ReasoningDataQueryEngine
from .package_metadata_field import PackageMetadataField
from .rag_resource import RAGResource
from .rag_resource_collection import RAGResourceCollection
from ..llm import GeminiLLM
from ..common import log_debug
from huggingface_hub import snapshot_download

import pandas as pd
from typing import Optional
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


DEBUG = True

# default config
FETCH_K = 12
TOP_K = 5 
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
DOCUMENT_SEPARATOR = "\n---\n"

from typing import Any
import math
def is_value_present(x: Any):
    """Check if the value is present and not NaN"""
    return (
        x is not None
        and x != ""
        and x != "nan"
        and x != "None"
        and (not isinstance(x, float) or not math.isnan(x))
    )

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
        persist_dir: str = "chroma_db",
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        document_separator: str = DOCUMENT_SEPARATOR,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_separator = document_separator
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[self.document_separator],
        )

        # Initialize components
        self.llm = GeminiLLM(
            model=model_name,
            temperature=temperature,
            # max_tokens=1024,
            api_key=GOOGLE_API_KEY,
        )
        self.simple_llm = GeminiLLM(
            model="gemini/gemini-1.5-flash",
            temperature=0.0,
            api_key=GOOGLE_API_KEY,
        )

        log_debug(f"[RAG] Initializing embeddings...")
        if not os.path.exists("embeddings_model"):
            snapshot_download(
                repo_id="sentence-transformers/all-mpnet-base-v2",
                local_dir="embeddings_model",
                local_dir_use_symlinks=False
            )
        self.embeddings = HuggingFaceEmbeddings(model_name="embeddings_model")

        log_debug(f"[RAG] Initializing reranker...")
        if not os.path.exists("reranker_model"):
            snapshot_download(
                repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
                local_dir="reranker_model",
                local_dir_use_symlinks=False
            )
        self.reranker = CrossEncoder("reranker_model")
        
        log_debug(f"[RAG] Initializing RAGResourceCollection...")
        self.rag_resource_collection = RAGResourceCollection(
            embeddings=self.embeddings,
            persist_dir=self.persist_dir,
        )

        log_debug(f"[RAG] Initializing package fields interpretation...")
        from .package_metadata_field import PACKAGE_FIELDS
        self.package_fields_interpretation = "\n".join(
            f'- "{f.field_name}" (kiểu dữ liệu {f.field_type}): {f.field_description}'
            for f in PACKAGE_FIELDS
        )

    def update_dataframe(self, df: pd.DataFrame, source: str):
        rag_resource = RAGResource(
            source=source,
            df=df,
            splitter=self.splitter,
        )
        self.rag_resource_collection.add_resources([rag_resource])
    
    def query_vectordb(self, query: str) -> str:
        """
        Trả về context gần nhất với query.
        Bạn cần tự nhúng thông tin chat history liên quan vào query nếu cần.
        """
        with self.rag_resource_collection.resource_engines_locked.read() as resource_engines:
            vectorstore = resource_engines.vectorstore
        
            # Use similarity_search_with_score to get top fetch_k candidates with their scores
            candidates_with_scores = vectorstore.similarity_search_with_score(query, k=self.fetch_k)
            
        if DEBUG:
            log_debug(f"[query_vectordb] [retriever] candidates count: {len(candidates_with_scores)}")
            log_debug("[query_vectordb] [retriever] candidates taken and scores:")
            for doc, sc in candidates_with_scores:
                log_debug(f"  {doc.metadata.get('service_code')} -> {sc:.4f}")
        
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
            log_debug("[query_vectordb] [reranker] top taken and scored:")
            for doc, sc in scored_sorted:
                log_debug(f"  {doc.metadata.get('service_code')} -> {sc:.4f}")
        
        # Check confidence threshold
        # top_score = float(scored_sorted[0][1]) if scored_sorted else 0.0
        # if top_score < self.confidence_threshold:
        #     raise Exception(f"Top score {top_score:.4f} below threshold {self.confidence_threshold}")
        
        # Build context from top_k documents (already sliced to top_k above)
        top_docs = [doc for doc, _ in scored_sorted]
        context_sections = []
        for d in top_docs:
            src = d.metadata.get("source", "unknown")
            code = d.metadata.get("service_code", "")
            context_sections.append(f"Nhà mạng: {src} ; tên gói cước: {code} ;\n{d.page_content}")
        
        context = "\n---\n".join(context_sections)
        context = (
            "Dưới đây liệt kê một số thông tin liên quan. Hãy sử dụng thông tin này để trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn.\n"
            + "Nếu không tìm thấy thông tin liên quan, cần hiểu rằng bạn không thể trả lời câu hỏi này.\n"
            + "\n" + context
        )
        return context
    
    def query_reasoning(self, chat_history: str, query: str) -> str:
        # Generate SQL-like query using LLM
        sql_query = self._get_reasoning_query(
            chat_history=chat_history,
            question=query,
        )
        
        if DEBUG:
            log_debug(f"[query_reasoning] Generated SQL: {sql_query}")
        
        if sql_query is None:
            raise Exception("LLM không thể sinh SQL query cho câu hỏi này")
        
        # Compile and execute query
        with self.rag_resource_collection.resource_engines_locked.read() as resource_engines:
            reasoning_engine = resource_engines.reasoning_engine
            try:
                query_object = reasoning_engine.compile(sql_query)
                result_table = reasoning_engine.apply(query_object)
            except Exception as e:
                raise Exception(f"Không thể compile hoặc execute query: {e}")
            
        if DEBUG:
            log_debug(f"[query_reasoning] Result table shape: {result_table}")
        
        if not result_table:
            raise Exception("Query execution trả về bảng rỗng")
        
        queryObjectInterpretation = query_object.interpret(entity_name="gói cước")
        context = f"""
            Đã xác định được ngữ cảnh phù hợp. Các gói cước mà người dùng mong muốn là: {queryObjectInterpretation}
            Cụ thể, dưới đây là tất cả các gói cước phù hợp với yêu cầu của người dùng. Không còn gói cước nào khác.


            {"\n---\n".join(
                self._convert_df_to_text(result_table.to_df())
            )}
        """

        return '\n'.join(line.strip() for line in context.split('\n'))
    
    def _convert_df_to_text(self, df: pd.DataFrame) -> list[str]:
        texts: list[str] = []
        for _, row in df.iterrows():
            row_data = list(
                (key, value)
                for (key, value) in row.items()
                if is_value_present(value)
            )
            content = (
                "\n".join((f"{key}: {value}" for key, value in row_data))
                + "\n---\n"
            )
            texts.append(content)
        
        return texts

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

        {self.package_fields_interpretation}

        Chú ý 1: Nếu người dùng hỏi dung lượng thì cần chọn các cột sau: "4G tốc độ tiêu chuẩn/ngày", "4G tốc độ cao/ngày", "4G tốc độ tiêu chuẩn/chu kỳ", "4G tốc độ cao/chu kỳ", "Chi tiết".
        Chú ý 2: Bạn phải luôn SELECT các cột sau trong mọi trường hợp: "Nhà mạng", "Mã dịch vụ", "Cú pháp đăng ký", "Giá (VNĐ)", "Chi tiết".
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
            SELECT "Mã dịch vụ", "Cú pháp đăng ký", "Giá (VNĐ)", "Chi tiết" WHERE "Chi tiết" CONTAINS "<các từ khóa trong câu hỏi của người dùng>"
        Nếu câu hỏi hoàn toàn nằm ngoài phạm vi những thông tin gói cước, sim thẻ... như trên thì trả về IMPOSSIBLE.

        Hãy nghiên cứu các ví dụ về truy vấn và câu trả lời dưới đây, từ đó trả lời truy vấn đầu vào được đưa ra ở cuối.

        Ví dụ 1:
        - Lịch sử chat: Không có
        - Câu hỏi: Gói cước nào có giá rẻ nhất?
        - Trả lời: SELECT "Nhà mạng", "Mã dịch vụ", "Giá (VNĐ)" WHERE "Giá (VNĐ)" REACHES MIN

        Ví dụ 2:
        - Lịch sử chat: Người dùng hỏi giá gói cước rẻ nhất, trợ lý ảo trả lời gói SD70 (70.000đ/tháng, 30GB) là rẻ nhất.
        - Câu hỏi: Làm thế nào để đăng ký gói đấy?
        - Trả lời: SELECT "Nhà mạng", "Mã dịch vụ", "Cú pháp đăng ký", "Chi tiết" WHERE "Mã dịch vụ" = "SD70"
        
        Ví dụ 3:
        - Lịch sử chat: Người dùng hỏi về cách kiểm tra số dư tài khoản, trợ lý ảo trả lời cần bấm *101# để kiểm tra.
        - Câu hỏi: Em ơi thế sao thuê bao của anh cứ tự trừ tiền thế nhỉ, em xem giúp anh số dư còn bao nhiêu với
        - Trả lời: IMPOSSIBLE
        
        Ví dụ 4:
        - Lịch sử chat: Không có
        - Câu hỏi: Ừ thế xem giúp anh gói nào để anh lướt mạng thả ga đi, một ngày xem phim đã tốn mấy gigabyte rồi
        - Trả lời: SELECT "Nhà mạng", "Mã dịch vụ", "4G tốc độ tiêu chuẩn/ngày" WHERE "Chi tiết" CONTAINS "lướt mạng thả ga" AND "4G tốc độ tiêu chuẩn/ngày" REACHES MIN
        
        Ví dụ 5:
        - Lịch sử chat: Không có
        - Câu hỏi: À em ơi bên em có gói nào rẻ mà lướt mạng thoải mái không, chứ một ngày anh lướt mạng hết mấy gigabyte rồi
        - Trả lời: SELECT "Nhà mạng", "Mã dịch vụ", "4G tốc độ tiêu chuẩn/ngày" WHERE "Chi tiết" CONTAINS "lướt mạng thoải mái" AND "4G tốc độ tiêu chuẩn/ngày" REACHES MIN AND "Giá (VNĐ)" REACHES MIN
        
        Ví dụ 6:
        - Lịch sử chat:
            Trợ lý ảo gợi ý gói 6MXH100 (180GB/tháng) và 12MXH100 (360GB/tháng) vì phù hợp nhu cầu xem phim nhiều.
            Người dùng hỏi cụ thể về ưu đãi của các gói 6MXH100 và 12MXH100, muốn tìm gói rẻ mà nhiều data.
            Trợ lý ảo gợi ý thêm các gói SD70 (70.000đ/tháng, 30GB), V90B (90.000đ/tháng, 30GB) và MXH100 (100.000đ/tháng, 30GB), lưu ý data có thể không đủ nếu xem phim nhiều.
            Người dùng muốn được tư vấn gói 70.000đ/tháng.
            Trợ lý ảo xác nhận gói SD70 (70.000đ/tháng, 30GB) phù hợp yêu cầu, nhưng lưu ý 30GB có thể không đủ cho nhu cầu xem phim nhiều. Gợi ý tham khảo các gói data lớn hơn nếu cần.
        - Câu hỏi: À vậy gói này đăng ký thế nào em nhỉ?
        - Trả lời: SELECT "Nhà mạng", "Mã dịch vụ", "Cú pháp đăng ký", "Giá (VNĐ)", "Chi tiết" WHERE "Mã dịch vụ" = "SD70"

        Ví dụ 7:
        - Lịch sử chat: Không có
        - Câu hỏi: Bên Viettel có gói cước nào rẻ nhất?
        - Trả lời: SELECT "Nhà mạng", "Mã dịch vụ", "Giá (VNĐ)" WHERE "Nhà mạng" = "Viettel" AND "Giá (VNĐ)" REACHES MIN

        ---
        
        Hãy trả lời truy vấn dưới đây:
        - Lịch sử chat: {chat_history}
        
        - Câu hỏi: {question}
        """
        
        if DEBUG:
            log_debug(f"[_get_reasoning_query] calling LLM...")
        
        prompt = prompt_temp.format(question=question)
        response = self.llm.call(prompt)
        
        if DEBUG:
            log_debug(f"[_get_reasoning_query] response: {response[:200]}...")
        
        response_text = response.strip()
        return None if "impossible" in response_text.lower() else response_text
