import os
from typing import List, Dict, Optional

from .chunker import chunk_text
from .embeddings import EmbeddingProvider
from .vector_store import FaissVectorStore

import logging

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, embedding_model: str | None = None, index_path: str | None = None, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedder = EmbeddingProvider(model=embedding_model)
        # try to infer dim by embedding a short text
        sample = self.embedder.embed(["hello"])
        dim = len(sample[0]) if sample else 384
        self.store = FaissVectorStore(index_path=index_path, dim=dim)

    def ingest_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)
        texts = [c["text"] for c in chunks]
        ids = [f"{doc_id}::{c['id']}" for c in chunks]
        metas = [{"doc_id": doc_id, "chunk_index": i, **(metadata or {})} for i, _ in enumerate(chunks)]
        embs = self.embedder.embed(texts)
        self.store.upsert(ids, embs, metas)
        logger.info("Ingested %d chunks for doc %s", len(ids), doc_id)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        qemb = self.embedder.embed([query])[0]
        hits = self.store.search_by_vector(qemb, top_k=top_k)
        contexts = []
        for meta, score in hits:
            ctx = {"meta": meta, "score": score}
            contexts.append(ctx)
        return contexts

    def answer(self, query: str, top_k: int = 5, call_llm: bool = False) -> Dict:
        contexts = self.retrieve(query, top_k=top_k)
        assembled = "\n\n".join([c["meta"].get("meta", {}).get("text", "") or "" for c in contexts])
        prompt = (
            "You are a helpful assistant for fund performance documents.\n"
            "Use the context below to answer the question. If the answer is not present, say you don't know.\n\n"
            "CONTEXT:\n"
            f"{assembled}\n\nQUESTION:\n{query}\n"
        )
        response = {"prompt": prompt, "contexts": contexts}
        if call_llm and os.getenv("OPENAI_API_KEY"):
            try:
                import openai

                openai.api_key = os.getenv("OPENAI_API_KEY")
                chat = openai.ChatCompletion.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                text = chat["choices"][0]["message"]["content"]
                response["answer"] = text
            except Exception as e:
                logger.exception("LLM call failed: %s", e)
                response["answer_error"] = str(e)
        return response