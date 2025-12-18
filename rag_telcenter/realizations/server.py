from ..engine import RAG
import pandas as pd
from ..services.MessageQueueService import MessageQueueService
from typing import Any, Callable
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
    def __init__(self, method_map: dict[str, Callable], mq: MessageQueueService, response_queue_name: str) -> None:
        self.mq = mq
        self.response_queue_name = response_queue_name
        self.method_map = method_map


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

        print(f"[Controller] Publishing message into queue {self.response_queue_name}: {response}")

        mq = self.mq.clone()
        mq.publish_message(self.response_queue_name, response)
        print(f"[Controller] Published message into queue {self.response_queue_name} ____________")
    
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
    def __init__(self, method_map: dict[str, Callable], request_queue_name: str, response_queue_name: str) -> None:
        self.mq_service = MessageQueueService()
        self.mq_lock = threading.Lock()

        self.request_queue_name = request_queue_name
        self.response_queue_name = response_queue_name
        self.method_map = method_map
        
        self.threads: list[threading.Thread] = []
        self.num_threads = 4
    
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
        controller = Controller(self.method_map, mq, self.response_queue_name)
        mq.register_callback(self.request_queue_name, controller.handle_message)
        mq.start_consuming()
        


def main():
    rag_rpc_server = RAGRPCServer()

    rag_query_server = Server(
        method_map={
            "query_vectordb": rag_rpc_server.query_vectordb,
            "query_reasoning": rag_rpc_server.query_reasoning,
        },
        request_queue_name="telcenter_rag_text_requests",
        response_queue_name="telcenter_rag_text_responses",
    )
    rag_query_server.start()

    rag_update_server = Server(
        method_map={
            "update_dataframe": rag_rpc_server.update_dataframe,
        },
        request_queue_name="telcenter_rag_update_requests",
        response_queue_name="telcenter_rag_update_responses",
    )
    rag_update_server.start()

    rag_query_server.wait()
    rag_update_server.wait()
