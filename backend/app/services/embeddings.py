import os
from typing import List

import numpy as np

# optional imports (import inside functions to avoid hard dependency at import time)


class EmbeddingProvider:
    """
    Provides embeddings using OpenAI (if OPENAI_API_KEY present) else sentence-transformers.
    """

    def __init__(self, model: str | None = None):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._st_model = None

    def _ensure_st(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.openai_key:
            try:
                import openai
            except Exception:
                self.openai_key = None  # fallback if openai not installed
            else:
                openai.api_key = self.openai_key
                # OpenAI supports batching
                resp = openai.Embedding.create(model=self.model, input=texts)
                embeds = [d["embedding"] for d in resp["data"]]
                return embeds

        # fallback to sentence-transformers
        self._ensure_st()
        embs = self._st_model.encode(texts, show_progress_bar=False)
        # ensure list[list[float]]
        return [list(map(float, e)) for e in np.asarray(embs)]