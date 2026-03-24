from __future__ import annotations

from dataclasses import dataclass

from ..models import AnalysisResult, ProfileRecord

_SUSPICIOUS_TERMS = {
    "leak",
    "breach",
    "fraud",
    "scam",
    "impersonation",
    "phishing",
}


@dataclass(slots=True)
class RuleBasedAnalyzer:
    """Current deterministic analyzer implementation."""

    base_score: float = 0.1

    def analyze(self, record: ProfileRecord) -> AnalysisResult:
        text = f"{record.display_name} {record.url} {record.metadata}".casefold()

        hits = tuple(sorted(term for term in _SUSPICIOUS_TERMS if term in text))
        score = min(1.0, self.base_score + 0.2 * len(hits))

        if hits:
            rationale = f"Matched suspicious terms: {', '.join(hits)}"
            tags = ("risk",) + hits
        else:
            rationale = "No suspicious lexical indicators detected"
            tags = ("benign",)

        return AnalysisResult(risk_score=score, tags=tags, rationale=rationale)
