from ..engine import RAG
import pandas as pd
from ..services.MessageQueueService import MessageQueueService
from typing import Any
import threading

class RAGRPCServer:
    def __init__(self) -> None:
        self.rag = RAG()
    
    def update_dataframe(self, df: dict[str, Any], source: str) -> str:
        pd_df = pd.DataFrame.from_dict(df)
        self.rag.update_dataframe(pd_df, source=source)
        return "OK"
    
    def query_vectordb(self, query: str) -> str:
        context = self.rag.query_vectordb(query=query)
        return context
    
    def query_reasoning(self, chat_history: str, query: str) -> str:
        context = self.rag.query_reasoning(
            chat_history=chat_history,
            query=query,
        )
        return context

class Controller:
    def __init__(self, mq: MessageQueueService, response_queue_name: str, rag_rpc_server: RAGRPCServer) -> None:
        self.mq = mq
        self.response_queue_name = response_queue_name
        self.rag_rpc_server = rag_rpc_server
        self.method_map = {
            "update_dataframe": self.rag_rpc_server.update_dataframe,
            "query_vectordb": self.rag_rpc_server.query_vectordb,
            "query_reasoning": self.rag_rpc_server.query_reasoning,
        }


    def handle_message(self, message: dict):
        id = message.get("id", None)
        if id is None or not isinstance(id, str):
            return # ignore messages without valid id
        
        result_status = "success"
        try:
            result_content = self.handle_message_with_id(id, message)
        except Exception as e:
            result_status = "error"
            result_content = { "message": str(e) }
        
        response = {
            "id": id,
            "result": {
                "status": result_status,
                "content": result_content,
            }
        }

        self.mq.publish_message(self.response_queue_name, response)
    
    def handle_message_with_id(self, id: str, message: dict):
        method_name = message.get("method", "")
        if not method_name:
            raise ValueError("Message missing 'method' field")
        
        method = self.method_map.get(method_name, None)
        if method is None:
            raise ValueError(f"Unknown method: {method_name}")

        params = message.get("params", {})
        args = []
        kwargs = {}
        if isinstance(params, dict):
            kwargs = params
        elif isinstance(params, list):
            args = params
        else:
            raise ValueError("'params' field must be a dict or list")
        
        return method(*args, **kwargs)

class Server:
    def __init__(self) -> None:
        self.mq_service = MessageQueueService()
        self.mq_lock = threading.Lock()
        self.request_queue_name = "telcenter_rag_text_requests"
        self.response_queue_name = "telcenter_rag_text_responses"
        self.threads: list[threading.Thread] = []
        self.num_threads = 4
        self.rag_rpc_server = RAGRPCServer()
    
    def start(self):
        self.threads = [
            threading.Thread(target=self._consume_in_background, daemon=True)
            for _ in range(self.num_threads)
        ]
        for t in self.threads:
            t.start()

    def wait(self):
        for t in self.threads:
            t.join()

    def _consume_in_background(self):
        with self.mq_lock:
            mq = self.mq_service.clone()
        mq.declare_queue(self.request_queue_name)
        mq.declare_queue(self.response_queue_name)
        controller = Controller(mq, self.response_queue_name, self.rag_rpc_server)
        mq.register_callback(self.request_queue_name, controller.handle_message)
        mq.start_consuming()
        


def main():
    server = Server()
    server.start()
    server.wait()
