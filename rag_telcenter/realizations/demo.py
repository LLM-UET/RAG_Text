from ..engine import RAG
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def main():
    # Initialize RAG system
    rag = RAG()
    
    # Load data and update RAG
    df = load_csv("viettel.csv")
    rag.update_dataframe(df, source="Viettel")
    
    # Test queries
    queries = [
        # "Để đăng ký gói MXH120 thì soạn tin gửi 191 đúng không?",
        # "Nêu cú pháp đăng ký của gói cước có giá 120000 VNĐ và có ưu đãi miễn phí data",
        # "Liệt kê tất cả các gói cước có chu kỳ lớn hơn 30 ngày?",
        # "Bạn bao nhiêu tuổi?",
        "Gói cước nào không giới hạn dung lượng mạng mà có giá rẻ dưới 150.000 VNĐ?",
    ]
    
    for q in queries:
        print("\n" + "="*80)
        print(f"QUESTION: {q}")
        print("-"*80)

        print(f"=" * 80)
        print(f"Trying vectorstore RAG...")
        print(f"=" * 80)
        # vectordb
        try:
            context = rag.query_vectordb(query=q) # TODO: include chat history
            print(f"[VECTORDB] Success!")
            print(f"Context preview:\n{context[:500]}...")
        except Exception as e2:
            print(f"[VECTORDB] Failed: {e2}")

        print(f"=" * 80)
        print(f"Trying reasoning RAG...")
        print(f"=" * 80)
        # reasoning
        try:
            context = rag.query_reasoning(
                chat_history="",
                query=q,
            )
            print(f"[REASONING] Success!")
            print(f"Result table:\n{context[:500]}...")
        except Exception as e:
            print(f"[REASONING] Failed: {e}")

            # TODO: Who is Partner?
            print("Answer: Xin lỗi, tôi không thể trả lời câu hỏi này. Bạn có muốn tôi chuyển tiếp tới tư vấn viên tổng đài của partner để được hỗ trợ không?")

        print(f"=" * 80)
        print(f"DONE.")
        print(f"=" * 80)
