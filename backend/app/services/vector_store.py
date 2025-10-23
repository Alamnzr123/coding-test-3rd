"""
Vector store service using pgvector (PostgreSQL extension)

TODO: Implement vector storage using pgvector
- Create embeddings table in PostgreSQL
- Store document chunks with vector embeddings
- Implement similarity search using pgvector operators
- Handle metadata filtering
"""
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings
from app.db.session import SessionLocal


class VectorStore:
    """pgvector-based vector store for document embeddings"""
    
    def __init__(self, db: Session = None):
        self.db = db or SessionLocal()
        self.embeddings = self._initialize_embeddings()
        self._ensure_extension()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if settings.OPENAI_API_KEY:
            return OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )
        else:
            # Fallback to local embeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled
        
        TODO: Implement this method
        - Execute: CREATE EXTENSION IF NOT EXISTS vector;
        - Create embeddings table if not exists
        """
        try:
            # Enable pgvector extension
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # Create embeddings table
            # Dimension: 1536 for OpenAI, 384 for sentence-transformers
            dimension = 1536 if settings.OPENAI_API_KEY else 384
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER,
                fund_id INTEGER,
                content TEXT NOT NULL,
                embedding vector({dimension}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx 
            ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
            
            self.db.execute(text(create_table_sql))
            self.db.commit()
        except Exception as e:
            print(f"Error ensuring pgvector extension: {e}")
            self.db.rollback()
    
    async def add_document(self, content: str, metadata: Dict[str, Any]):
        """
        Add a document to the vector store
        
        TODO: Implement this method
        - Generate embedding for content
        - Insert into document_embeddings table
        - Store metadata as JSONB
        """
        try:
            # Generate embedding
            embedding = await self._get_embedding(content)
            embedding_list = embedding.tolist()
            
            # Insert into database
            insert_sql = text("""
                INSERT INTO document_embeddings (document_id, fund_id, content, embedding, metadata)
                VALUES (:document_id, :fund_id, :content, :embedding::vector, :metadata::jsonb)
            """)
            
            self.db.execute(insert_sql, {
                "document_id": metadata.get("document_id"),
                "fund_id": metadata.get("fund_id"),
                "content": content,
                "embedding": str(embedding_list),
                "metadata": str(metadata)
            })
            self.db.commit()
        except Exception as e:
            print(f"Error adding document: {e}")
            self.db.rollback()
            raise
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity
        
        TODO: Implement this method
        - Generate query embedding
        - Use pgvector's <=> operator for cosine distance
        - Apply metadata filters if provided
        - Return top k results
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"fund_id": 1})
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = await self._get_embedding(query)
            embedding_list = query_embedding.tolist()
            
            # Build query with optional filters
            where_clause = ""
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    if key in ["document_id", "fund_id"]:
                        conditions.append(f"{key} = {value}")
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)
            
            # Search using cosine distance (<=> operator)
            search_sql = text(f"""
                SELECT 
                    id,
                    document_id,
                    fund_id,
                    content,
                    metadata,
                    1 - (embedding <=> :query_embedding::vector) as similarity_score
                FROM document_embeddings
                {where_clause}
                ORDER BY embedding <=> :query_embedding::vector
                LIMIT :k
            """)
            
            result = self.db.execute(search_sql, {
                "query_embedding": str(embedding_list),
                "k": k
            })
            
            # Format results
            results = []
            for row in result:
                results.append({
                    "id": row[0],
                    "document_id": row[1],
                    "fund_id": row[2],
                    "content": row[3],
                    "metadata": row[4],
                    "score": float(row[5])
                })
            
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if hasattr(self.embeddings, 'embed_query'):
            embedding = self.embeddings.embed_query(text)
        else:
            embedding = self.embeddings.encode(text)
        
        return np.array(embedding, dtype=np.float32)
    
    def clear(self, fund_id: Optional[int] = None):
        """
        Clear the vector store
        
        TODO: Implement this method
        - Delete all embeddings (or filter by fund_id)
        """
        try:
            if fund_id:
                delete_sql = text("DELETE FROM document_embeddings WHERE fund_id = :fund_id")
                self.db.execute(delete_sql, {"fund_id": fund_id})
            else:
                delete_sql = text("DELETE FROM document_embeddings")
                self.db.execute(delete_sql)
            
            self.db.commit()
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            self.db.rollback()


# FAISS-backed simple vector store with metadata
class FaissVectorStore:
    def __init__(self, index_path: str | None = None, dim: int = 384):
        self.index_path = Path(index_path or os.getenv("FAISS_INDEX_PATH", "/app/faiss_index"))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.index_path.with_suffix(".meta.json")
        self.dim = dim
        self._index = None
        self._metadict: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        try:
            import faiss
        except Exception:
            self._index = None
            return
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as fh:
                    self._metadict = json.load(fh)
        else:
            self._index = faiss.IndexFlatIP(self.dim)

    def _save(self):
        if self._index is None:
            return
        import faiss

        faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._metadict, fh, ensure_ascii=False, indent=2)

    def upsert(self, ids: List[str], embeddings: List[List[float]], metas: List[Dict]):
        """
        ids: list of string ids
        embeddings: list of vectors (matching dim)
        metas: list of metadata dicts
        """
        try:
            import faiss
        except Exception as e:
            raise RuntimeError("faiss not available; install faiss-cpu or use pgvector") from e

        vecs = np.array(embeddings).astype("float32")
        if vecs.shape[1] != self.dim:
            # allow adaptable dim for first time
            if self._index is None:
                self.dim = vecs.shape[1]
                self._index = faiss.IndexFlatIP(self.dim)
            else:
                raise ValueError("Embedding dim mismatch")

        self._index.add(vecs)
        # store metadata keyed by incremental numeric idx (simple approach)
        start_idx = len(self._metadict)
        for i, _id in enumerate(ids):
            self._metadict[str(start_idx + i)] = {"id": _id, "meta": metas[i]}
        self._save()

    def search_by_vector(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict, float]]:
        try:
            import faiss
        except Exception:
            raise RuntimeError("faiss not available")
        if self._index is None:
            return []
        q = np.array([query_embedding]).astype("float32")
        D, I = self._index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = self._metadict.get(str(int(idx)), {})
            results.append((meta, float(score)))
        return results

    def close(self):
        self._save()
