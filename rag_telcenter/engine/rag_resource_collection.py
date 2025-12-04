from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import pandas as pd
import numpy as np

from .rag_resource import RAGResource
from .rag_reasoning import ReasoningDataQueryEngine
from ..common import RWResource

from dataclasses import dataclass
@dataclass(frozen=True)
class RAGResourceEngines:
    """Immutable"""
    vectorstore: Chroma
    reasoning_engine: ReasoningDataQueryEngine


class RAGResourceCollection:
    def __init__(self, embeddings: Embeddings, persist_dir: str | None):
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self.resource_engines_locked = RWResource(
            RAGResourceEngines(
                vectorstore=Chroma(embedding_function=embeddings, persist_directory=persist_dir),
                reasoning_engine=ReasoningDataQueryEngine(df=pd.DataFrame(), embedder=lambda x: np.array(self.embeddings.embed_query(x))),
            )
        )

        d: dict[str, RAGResource] = {}
        self.source_to_resource_locked = RWResource(d)
    
    def add_resources(self, resources: list[RAGResource]):
        with self.source_to_resource_locked.write() as w:
            for resource in resources:
                w.value[resource.source] = resource

        self._rebuild_indexes()
    
    def _rebuild_indexes(self):
        chunks: list[Document] = []
        df = pd.DataFrame()

        with self.source_to_resource_locked.read() as source_to_resource:
            for resource in source_to_resource.values():
                chunks.extend(resource.chunks)
                df = pd.concat([df, resource.df], ignore_index=True)

        reasoning_engine = ReasoningDataQueryEngine(
            df=df,
            embedder=lambda x: np.array(self.embeddings.embed_query(x)),
        )

        with self.resource_engines_locked.write() as w:
            vectorstore = w.value.vectorstore

            doc_ids: list[str] = []
            for doc in chunks:
                if doc.id is None:
                    raise ValueError("Document id is None")
                doc_ids.append(doc.id)
                parent_id = doc.metadata.get("parent_id", None)
                if parent_id:
                    vectorstore._collection.delete(where={"parent_id": parent_id})

            vectorstore._collection.upsert(
                ids=doc_ids,
                documents=[doc.page_content for doc in chunks],
                embeddings=self.embeddings.embed_documents([
                    doc.page_content + "\n" + str(doc.metadata) + "\n" + str(doc.id) for doc in chunks
                ]), # type: ignore
                metadatas=[doc.metadata for doc in chunks],
            )

            w.value = RAGResourceEngines(
                vectorstore=vectorstore,
                reasoning_engine=reasoning_engine,
            )
