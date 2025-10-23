"""
Query engine service for RAG-based question answering
"""
import os
import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings
from app.services.vector_store import VectorStore
from app.services.metrics_calculator import MetricsCalculator
from sqlalchemy.orm import Session
from app.db.session import SessionLocal



logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Lightweight rule-based intent classifier.
    Returns one of: "calculation", "definition", "retrieval", "conversation", "unknown"
    """
    CALC_KEYWORDS = {"calculate", "what is", "compute", "irr", "dpi", "pic", "paid-in", "payment", "how much"}
    DEF_KEYWORDS = {"what is", "define", "meaning of", "definition", "means"}
    RETRIEVE_KEYWORDS = {"show", "list", "give me", "all", "transactions", "capital calls", "distributions", "adjustments"}
    CONV_KEYWORDS = {"hello", "hi", "thanks", "thank you"}

    @classmethod
    def classify(cls, text: str) -> Tuple[str, float]:
        t = text.lower()
        score = {"calculation": 0, "definition": 0, "retrieval": 0, "conversation": 0}
        for kw in cls.CALC_KEYWORDS:
            if kw in t:
                score["calculation"] += 2
        for kw in cls.DEF_KEYWORDS:
            if kw in t:
                score["definition"] += 2
        for kw in cls.RETRIEVE_KEYWORDS:
            if kw in t:
                score["retrieval"] += 2
        for kw in cls.CONV_KEYWORDS:
            if kw in t:
                score["conversation"] += 1

        best = max(score, key=score.get)
        total = sum(score.values()) or 1
        confidence = float(score[best]) / float(total)
        if score[best] == 0:
            return "unknown", 0.0
        return best, confidence


class LLMWrapper:
    """
    Minimal wrapper: prefer OpenAI ChatCompletion, fall back to LangChain HuggingFaceHub (if configured).
    Returns text output string.
    """
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def chat(self, prompt: str, max_tokens: int = 512) -> str:
        if self.openai_key:
            try:
                import openai
                openai.api_key = self.openai_key
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.exception("OpenAI call failed, falling back: %s", e)

        if self.hf_token:
            try:
                from langchain import HuggingFaceHub
                llm = HuggingFaceHub(
                    repo_id=os.getenv("HUGGINGFACE_REPO_ID", "google/flan-t5-small"),
                    huggingfacehub_api_token=self.hf_token
                )
                return llm(prompt)
            except Exception as e:
                logger.exception("HuggingFaceHub call failed: %s", e)

        # final fallback: simple echo / safety
        return "Sorry, LLM not configured. Install OPENAI_API_KEY or HUGGINGFACE_API_TOKEN."


class ConversationManager:
    """
    Simple in-memory conversation store. Replace with DB for production.
    """
    def __init__(self):
        self._store: Dict[str, List[Dict]] = {}

    def add_message(self, conv_id: str, role: str, text: str):
        self._store.setdefault(conv_id, []).append({"role": role, "text": text})

    def history(self, conv_id: str) -> List[Dict]:
        return self._store.get(conv_id, [])

    def clear(self, conv_id: str):
        if conv_id in self._store:
            del self._store[conv_id]


class ResponseFormatter:
    @staticmethod
    def format_answer(answer: str, sources: Optional[List[Dict]] = None, calc: Optional[Dict] = None) -> Dict:
        res = {"answer": answer}
        if sources:
            # normalize sources: include page/doc/chunk metadata if present
            formatted = []
            for s in sources:
                meta = s.get("meta") if isinstance(s, dict) else s
                formatted.append({
                    "score": float(s.get("score", 0)) if isinstance(s, dict) else 0.0,
                    "meta": meta
                })
            res["sources"] = formatted
        if calc:
            res["calculation"] = calc
        return res


class QueryEngine:
    """
    High-level orchestrator:
     - classifies intent
     - routes to metrics calculator or RAG retrieval
     - calls LLM for generation when needed
    """
    def __init__(self, db: Optional[Session] = None):
        self.intent = IntentClassifier()
        self.llm = LLMWrapper()
        self.vector_store = VectorStore()
        # Do not create MetricsCalculator here (needs DB). Instantiate per-call.
        self.metrics_cls = MetricsCalculator
        self.conv = ConversationManager()
        # optional DB session passed from caller (chat endpoint passes db)
        self.db = db

    def handle_query(self, query: str, fund_id: Optional[int] = None,
                     conv_id: Optional[str] = None, top_k: int = 5, call_llm: bool = True,
                     db: Optional[Session] = None) -> Dict:
        intent, conf = self.intent.classify(query)
        logger.info("Query intent=%s conf=%.2f q=%s", intent, conf, query)

        # conversation context append
        if conv_id:
            self.conv.add_message(conv_id, "user", query)

        if intent == "calculation":
            # route to metrics calculator
            try:
                created_session = False
                if db is None:
                    db = SessionLocal()
                    created_session = True
                metrics = self.metrics_cls(db)
                calc_res = self._handle_calculation(query, fund_id, metrics)
                answer = calc_res.get("human_readable", calc_res)
                formatted = ResponseFormatter.format_answer(answer, sources=None, calc=calc_res)
                if conv_id:
                    self.conv.add_message(conv_id, "assistant", answer)
                return {"intent": intent, "confidence": conf, "result": formatted}
            except Exception as e:
                logger.exception("Calculation failed: %s", e)
                return {"intent": intent, "confidence": conf, "error": str(e)}
            finally:
                if 'created_session' in locals() and created_session:
                    try:
                        db.close()
                    except Exception:
                        pass

        if intent in ("retrieval", "definition", "unknown"):
            # run RAG retrieval
            contexts = self.vector_store.similarity_search(
                query=query,
                k=top_k,
                filter_metadata={"fund_id": fund_id} if fund_id else None
            )
            prompt = self._build_prompt(query, contexts)
            llm_answer = ""
            if call_llm:
                llm_answer = self.llm.chat(prompt)
            else:
                llm_answer = " ".join([c.get("meta", {}).get("meta", "") for c in contexts]) or "No context found."

            formatted = ResponseFormatter.format_answer(llm_answer, sources=contexts)
            if conv_id:
                self.conv.add_message(conv_id, "assistant", llm_answer)
            return {"intent": intent, "confidence": conf, "result": formatted}

        # conversation
        if intent == "conversation":
            reply = "Hi — how can I help with fund metrics or documents?"
            if conv_id:
                self.conv.add_message(conv_id, "assistant", reply)
            return {"intent": intent, "confidence": conf, "result": ResponseFormatter.format_answer(reply)}

    def _handle_calculation(self, query: str, fund_id: Optional[int], metrics: MetricsCalculator) -> Dict:
        """
        Very small natural-language mapping to known calculations.
        Examples supported: DPI, IRR, PIC
        """
        q = query.lower()
        if "dpi" in q or "distribution to paid-in" in q:
            res = metrics.calculate_dpi(fund_id)
            return {"type": "dpi", "value": res, "human_readable": f"DPI = {res:.4f}"}
        if "irr" in q or "internal rate" in q:
            res = metrics.calculate_irr(fund_id)
            return {"type": "irr", "value": res, "human_readable": f"IRR = {res:.4%}"}
        if "pic" in q or "paid-in" in q:
            res = metrics.calculate_pic(fund_id)
            return {"type": "pic", "value": res, "human_readable": f"PIC = {res:.2f}"}
        # fallback: try generic metrics endpoint
        return {"type": "unknown_calc", "value": None, "human_readable": "I couldn't map the requested calculation."}

    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        assembled = "\n\n".join([c.get("meta", {}).get("meta", "") or "" for c in contexts])
        prompt = (
            "You are an assistant specialized in private equity fund performance. Use the context below to answer.\n\n"
            f"CONTEXT:\n{assembled}\n\nQUESTION:\n{query}\n\n"
            "When you use facts from the context, cite the source meta (doc_id/chunk_index) at the end in a SOURCES section."
        )
        return prompt

    async def process_query(self, query: str, fund_id: Optional[int] = None, conversation_history: Optional[list] = None):
        """
        Async wrapper used by the chat endpoint. Runs the synchronous handle_query in a thread,
        normalizes result to {"answer": str, "sources": [...], "raw": ...} so existing callers
        (chat endpoint) can access response["answer"].
        """
        import asyncio

        # pass DB session into handle_query when present
        res = await asyncio.to_thread(
            self.handle_query,
            query,
            fund_id,
            None,             # conv_id (we keep conversation_history out-of-band)
            5,                # top_k default
            True,             # call_llm
            self.db
        )

        # normalize to expected chat response shape
        answer = ""
        sources = []
        if isinstance(res, dict):
            # calculation path returns {"intent", "confidence", "result": formatted}
            if "result" in res:
                result = res["result"]
                if isinstance(result, dict):
                    answer = result.get("answer") or result.get("human_readable") or str(result)
                    sources = result.get("sources") or []
                else:
                    answer = str(result)
            # fallback: if handle_query returned an 'answer' directly
            elif "answer" in res:
                answer = res["answer"]
            else:
                answer = str(res)
        else:
            answer = str(res)

        return {"answer": answer, "sources": sources, "raw": res}
