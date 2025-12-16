DEBUG = True

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
import pandas as pd
from .package_metadata_field import PACKAGE_FIELDS

class RAGResource:
    @staticmethod
    def _df_to_documents(df: pd.DataFrame, source: str) -> List[Document]:
        docs: list[Document] = []
        for _, row in df.iterrows():
            code = str(row.get("Mã dịch vụ", "")).strip()
            text_parts = []
            for col in df.columns:
                val = row[col]
                text_parts.append(f"{col}: {val}")
            text = " . ".join(text_parts)
            doc_id = f"{source}-{code}"
            doc = Document(
                page_content=text,
                metadata={"source": source, "service_code": code, "id": doc_id},
                id=doc_id,
            )
            doc.id = doc_id
            docs.append(doc)
        return docs
    



    @staticmethod
    def _chunk_documents(splitter: TextSplitter, docs: List[Document]) -> List[Document]:
        chunks = splitter.split_documents(docs)
        doc_id_counter: dict[str, int] = {}
        for chunk in chunks:
            base_id = chunk.metadata.get("id", "unknown")
            count = doc_id_counter.get(base_id, 0)
            
            chunk.id = f"{base_id}-chunk-{count}"
            chunk.metadata["id"] = chunk.id
            chunk.metadata["parent_id"] = base_id

            doc_id_counter[base_id] = count + 1
        return chunks



    def __init__(
        self,
        source: str,
        df: pd.DataFrame,
        splitter: TextSplitter,
    ):
        self.source = source
        self.df = df.copy()
        self.df.columns = [c.strip() for c in self.df.columns]
        self.df = self.df.fillna("")

        self.df =self.normalize_inplace(self.df)

        docs = self._df_to_documents(self.df, source)
        chunks = self._chunk_documents(splitter, docs)
        if DEBUG:
            print(f"[RAGResource] New resource | from source {source} | having dataframe with {len(self.df)} rows, {len(docs)} docs, {len(chunks)} chunks")
        # self.docs = docs
        self.chunks = chunks
        self.df['Nhà mạng'] = source
    
    @staticmethod
    def normalize_inplace(df: pd.DataFrame) -> pd.DataFrame:
        fields = PACKAGE_FIELDS

        for field in fields:
            field_name = field.field_name.strip()
            if field_name in df.columns:
                column = field_name
                if field.field_type == "number":
                    df[column] = df[column].apply(
                        lambda x: float(str(x).strip()) if str(x).strip() != "" else 0
                    )
                    df[column] = df[column].astype(float)
                else:
                    df[column] = df[column].apply(
                        lambda x: str(x).strip() if str(x).strip() != "" else "không có"
                    )
                    df[column] = df[column].astype(str)
                df.rename(columns={column: field.field_name}, inplace=True)
            else:
                # add missing column
                if field.field_type == "number":
                    df[field.field_name] = 0.0
                    df[field.field_name] = df[field.field_name].astype(float)
                else:
                    df[field.field_name] = "không có"
                    df[field.field_name] = df[field.field_name].astype(str)
        
        return df
