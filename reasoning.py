# /d:/Project/RAG_Demo/reasoning.py

from typing import Optional, List, Any
from pydantic import BaseModel
import json
import re

# Try to import the LLM adapter; keep optional to allow testing/mocking
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore


class ReasoningDataQuery(BaseModel):
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    require_free_data: Optional[bool] = None
    require_daily_data: Optional[bool] = None
    cycle_min: Optional[int] = None
    cycle_max: Optional[int] = None
    keywords: Optional[List[str]] = None


class ReasoningDataQueryEngine:
    """
    Engine to parse a user question into a ReasoningDataQuery using a reasoning LLM.
    """

    SYSTEM_PROMPT = """
Bạn là bộ phân tích câu hỏi cho hệ thống RAG gói cước Viettel.
Nhiệm vụ: đọc câu hỏi người dùng và xuất ra JSON chứa các trường lọc dữ liệu.

JSON schema yêu cầu:
- price_min (int hoặc null)
- price_max (int hoặc null)
- require_free_data (bool hoặc null)
- require_daily_data (bool hoặc null)
- cycle_min (int hoặc null)
- cycle_max (int hoặc null)
- keywords (list string hoặc null)

Quy tắc:
- Nếu người dùng nói "miễn phí data", "free data" → require_free_data = true.
- Nếu nói "120k", "120.000", "120000" → price_min = price_max = 120000.
- Nếu nói "rẻ nhất", "giá thấp", "gói rẻ" → price_max = 30000 (default gói rẻ).
- Nếu nói "chu kỳ 30 ngày" → cycle_min = cycle_max = 30.
- Nếu không rõ → để null.
- keywords: chứa các từ đặc trưng như tên gói (VD: "MXH120"), các dịch vụ (FB, TikTok…).

Chỉ trả về JSON hợp lệ — không giải thích thêm.
"""

    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.0, llm: Any = None):
        # Allow injecting a mock llm for testing; otherwise create the default adapter if available
        if llm is not None:
            self.llm = llm
        else:
            if ChatGoogleGenerativeAI is None:
                raise RuntimeError("ChatGoogleGenerativeAI not available; provide llm parameter for testing.")
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    @staticmethod
    def _extract_json_fragment(text: str) -> Optional[str]:
        """
        Find the first JSON object or array in text and return it.
        """
        # Try to find a {...} block
        obj_match = re.search(r"(\{.*?\})", text, flags=re.DOTALL)
        if obj_match:
            return obj_match.group(1).strip()
        # Fallback: find a [...] block
        arr_match = re.search(r"(\{.*?\})", text, flags=re.DOTALL)
        if arr_match:
            return arr_match.group(1).strip()
        return None

    def extract_query(self, question: str) -> ReasoningDataQuery:
        """
        Send the combined prompt to the LLM and parse returned JSON into ReasoningDataQuery.
        On any parse failure, returns an empty ReasoningDataQuery (all fields None).
        """
        user_prompt = f'\nCâu hỏi: "{question}"\n\nHãy trả về JSON duy nhất tuân thủ đúng schema.\n'
        # Invoke the LLM. The adapter may return different shapes; handle common ones.
        out = self.llm.invoke(self.SYSTEM_PROMPT + user_prompt)

        # Normalize to text
        text = ""
        try:
            if isinstance(out, str):
                text = out.strip()
            elif hasattr(out, "content"):
                text = getattr(out, "content").strip()  # typical single-response object
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                first = out[0]
                if isinstance(first, str):
                    text = first.strip()
                elif hasattr(first, "content"):
                    text = getattr(first, "content").strip()
            elif isinstance(out, dict) and "content" in out:
                text = out["content"].strip()
            else:
                text = str(out).strip()
        except Exception:
            text = str(out).strip()

        # Extract JSON fragment (handles code fences and extra commentary)
        json_fragment = self._extract_json_fragment(text) or text

        try:
            data = json.loads(json_fragment)
            # Ensure keys exist and types are compatible via pydantic model
            return ReasoningDataQuery(**data)
        except Exception:
            # Fallback: return empty query (all None)
            return ReasoningDataQuery()
