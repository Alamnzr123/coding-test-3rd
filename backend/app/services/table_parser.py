import re
from typing import List, Tuple

import pandas as pd


class TableParser:
    """
    Lightweight rule-based table classifier for fund reports.
    """

    CAPITAL_KEYWORDS = {"capital", "call", "paid-in", "paid in", "pic", "contribution"}
    DISTRIBUTION_KEYWORDS = {"distribution", "distribute", "return", "dpi", "dividend"}
    ADJUSTMENT_KEYWORDS = {"adjust", "adjustment", "rebalance", "recall", "clawback", "refund"}

    @classmethod
    def headers_text(cls, df: pd.DataFrame) -> str:
        try:
            headers = [str(h).lower() for h in df.columns]
            return " ".join(headers)
        except Exception:
            return ""

    @classmethod
    def _score_keywords(cls, text: str, keywords: set) -> int:
        score = 0
        for kw in keywords:
            if kw in text:
                score += 2
            else:
                # fuzzy contains
                for token in text.split():
                    if kw in token or token in kw:
                        score += 1
        return score

    @classmethod
    def classify_with_confidence(cls, df: pd.DataFrame) -> Tuple[str, float]:
        text = cls.headers_text(df)
        tokens = set(re.findall(r"[a-z0-9\-]+", text.lower()))

        cap_score = cls._score_keywords(text, cls.CAPITAL_KEYWORDS)
        dist_score = cls._score_keywords(text, cls.DISTRIBUTION_KEYWORDS)
        adj_score = cls._score_keywords(text, cls.ADJUSTMENT_KEYWORDS)

        scores = {"capital_calls": cap_score, "distributions": dist_score, "adjustments": adj_score}
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        confidence = float(scores[best]) / float(total)

        # if scores are all zero, fallback to unknown
        if all(v == 0 for v in scores.values()):
            return "unknown", 0.0

        # small heuristics: if header contains 'date' and amounts, prefer unknown if low confidence
        if "date" in text and confidence < 0.35:
            return "unknown", confidence

        return best, confidence

    @classmethod
    def classify(cls, df: pd.DataFrame) -> str:
        name, _ = cls.classify_with_confidence(df)
        return name