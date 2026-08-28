"""Quality Gate — Core quality control engine for AI outputs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality assessment levels."""
    OK = "ok"
    WARNING = "warning"
    FAIL = "fail"


class QualityIssue(Enum):
    """Types of quality issues detected."""
    SELF_SCORE_LOW = "self_score_low"
    HALLUCINATION = "hallucination"
    MISSING_CITATIONS = "missing_citations"
    STRUCTURE_BROKEN = "structure_broken"
    HALLUCINATION_RATE = "hallucination_rate_high"


@dataclass
class QualityReport:
    """Detailed quality assessment report."""
    level: QualityLevel
    self_score: float  # 0-10
    hallucination_rate: float  # 0.0-1.0
    citation_count: int
    issues: list[QualityIssue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = "proceed"

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "self_score": self.self_score,
            "hallucination_rate": self.hallucination_rate,
            "citation_count": self.citation_count,
            "issues": [i.value for i in self.issues],
            "details": self.details,
            "recommendation": self.recommendation,
        }


def count_citations(text: str) -> int:
    """Count citations in text."""
    bracketed = len(re.findall(r'\[\d+\]', text))
    author_year = len(re.findall(r'\([A-Z][a-z]+,\s*\d{4}\)', text))
    return max(bracketed, author_year)


def check_internal_consistency(text: str) -> dict:
    """Check for internal contradictions."""
    contradictions = []
    # Pattern: "X est Y" AND "X n'est pas Y" / "X n'est plus Y" etc.
    pat_a = r"(?:le\s+\S+)\s+est\s+(?:pas\s+)?(?:un|une|de\s+la)\s+(\S+)"
    pat_b = r"(?:le\s+\S+)\s+n(?:est|pas|plus|jamais)\s+(?:un|une|de\s+la)"
    if re.search(pat_a, text) and re.search(pat_b, text):
        contradictions.append("contradictory_statements")
    return {
        "contradictions_found": len(contradictions),
        "details": contradictions,
    }


# Pre-compiled regex patterns for hallucination signals
_HALLUC_SIGNALS = {
    "fake_citations": re.compile(
        r"(?:Ref|Source|R[eé]f[eé]rence|Citation)\s+\d+:\s+\S+\s+—\s+\S+",
    ),
    "uncertain_markers": re.compile(
        r"(?:probablement|peut-être|apparemment|selon certaines sources|il semble)",
        re.IGNORECASE,
    ),
    "overconfident_facts": re.compile(
        r"(?:c'est prouvé|c'est certain|la vérité c'est que|"
        r"absolument impossible|tous les experts)",
        re.IGNORECASE,
    ),
    "made_up_names": re.compile(
        r"(?:Dr|Prof|Mme|M.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:a|ont|est)",
    ),
    "timestamp_anomalies": re.compile(
        r"(?:en\s+\d{4}|depuis\s+\d{4}|dans\s+\d+\s+ans)",
    ),
    "unclear_sources": re.compile(
        r"(?:d'après|selon une(?:source)?(?:non )?vérifiée|"
        r"sources multiples|divers rapports)",
        re.IGNORECASE,
    ),
}


def heuristic_hallucination_signals(text: str) -> dict:
    """Quick hallucination detection without external calls."""
    signals = {}
    for name, pattern in _HALLUC_SIGNALS.items():
        signals[name] = len(pattern.findall(text))

    consistency = check_internal_consistency(text)
    signals["internal_consistency"] = consistency
    return signals


def extract_claims_for_verification(text: str, prompt: str) -> list[dict]:
    """Extract factual claims from text for verification (heuristic)."""
    claims = []
    sentences = re.split(r'[.!?]+', text)

    number_re = re.compile(
        r'\d+[\.,]?\d*\s*(%|milliards|millions|euros|USD|dollars)'
    )
    name_re = re.compile(
        r'(?:Dr|Prof|Mme|M.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+'
    )
    date_re = re.compile(
        r"(?:en|depuis|dans|au cours de)\s+(?:janvier|février|mars|"
        r"avril|mai|juin|juillet|août|septembre|octobre|novembre|"
        r"décembre|20\d{2})",
        re.IGNORECASE,
    )

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        has_number = bool(number_re.search(sentence))
        has_name = bool(name_re.search(sentence))
        has_date = bool(date_re.search(sentence))

        if has_number or has_name or has_date:
            claims.append({
                "claim": sentence[:200],
                "type": "factual",
                "confidence": "medium",
                "has_number": has_number,
                "has_name": has_name,
                "has_date": has_date,
            })

    return claims


def heuristic_hallucination_rate(text: str, prompt: str) -> float:
    """Estimate hallucination rate using heuristic signals."""
    signals = heuristic_hallucination_signals(text)
    total_claims = len(extract_claims_for_verification(text, prompt))

    if total_claims == 0:
        return 0.0

    issues = 0
    max_issues = 0

    fake_cite = signals.get("fake_citations", 0)
    max_issues += 3
    if fake_cite > 0:
        issues += min(fake_cite * 2, 3)

    overconf = signals.get("overconfident_facts", 0)
    max_issues += 2
    if overconf > 0:
        issues += overconf

    uncertain = signals.get("uncertain_markers", 0)
    max_issues += 1
    if uncertain > 2:
        issues += 1

    consistency = signals.get("internal_consistency", {})
    contradictions = consistency.get("contradictions_found", 0)
    max_issues += 2
    issues += contradictions * 2

    rate = min(issues / max(max_issues, 1), 1.0)

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if total_claims > 0 and len(sentences) > 0:
        claim_ratio = total_claims / len(sentences)
        if claim_ratio > 0.5:
            rate += 0.1 * claim_ratio

    return min(rate, 1.0)


class QualityGate:
    """Main quality gate engine.

    Evaluates AI-generated text against configurable quality thresholds.
    Supports self-scoring, hallucination detection, and fallback strategies.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.self_score_threshold = self.config.get(
            "self_score_threshold", 6.0
        )
        self.hallucination_rate_threshold = self.config.get(
            "hallucination_rate_threshold", 0.15
        )
        self.warning_hallucination_threshold = self.config.get(
            "warning_hallucination_threshold", 0.08
        )
        self.min_citations = self.config.get("min_citations", 2)
        self.max_retries = self.config.get("max_retries", 2)
        self.fallback_responses = self.config.get("fallback_responses", True)

    def evaluate(
        self,
        text: str,
        prompt: str,
        check_hallucination: bool = True,
    ) -> QualityReport:
        """Evaluate text quality."""
        issues: list[QualityIssue] = []
        details: dict[str, Any] = {}

        # 1. Self-score
        self_score = self._self_score(text, prompt)
        details["self_score"] = self_score
        if self_score < self.self_score_threshold:
            issues.append(QualityIssue.SELF_SCORE_LOW)

        # 2. Hallucination detection
        hallucination_rate = 0.0
        if check_hallucination:
            hallucination_rate = heuristic_hallucination_rate(text, prompt)
            details["hallucination_rate"] = hallucination_rate
            details["hallucination_signals"] = heuristic_hallucination_signals(text)

            if hallucination_rate > self.hallucination_rate_threshold:
                issues.append(QualityIssue.HALLUCINATION_RATE)
            elif hallucination_rate > self.warning_hallucination_threshold:
                issues.append(QualityIssue.HALLUCINATION)

        # 3. Citation check
        citation_count = count_citations(text)
        details["citation_count"] = citation_count

        if self.config.get("require_citations_for_factual", True):
            claims = extract_claims_for_verification(text, prompt)
            if len(claims) > 0 and citation_count < self.min_citations:
                issues.append(QualityIssue.MISSING_CITATIONS)

        # 4. Determine level
        if len(issues) == 0:
            level = QualityLevel.OK
            recommendation = "proceed"
        elif len(issues) == 1:
            level = QualityLevel.WARNING
            recommendation = "regenerate" if QualityIssue.HALLUCINATION in issues else "proceed_with_caution"
        else:
            level = QualityLevel.FAIL
            recommendation = "fallback"

        # Very high hallucination -> forced FAIL
        if hallucination_rate > 0.5:
            level = QualityLevel.FAIL
            recommendation = "fallback"

        report = QualityReport(
            level=level,
            self_score=self_score,
            hallucination_rate=hallucination_rate,
            citation_count=citation_count,
            issues=issues,
            details=details,
            recommendation=recommendation,
        )

        logger.info(
            "Quality gate: level=%s self_score=%.1f hallucination_rate=%.1f%% citations=%d",
            level.value, self_score, hallucination_rate * 100, citation_count,
        )
        if issues:
            logger.warning("Quality issues: %s", [i.value for i in issues])

        return report

    def _self_score(self, text: str, prompt: str) -> float:
        """Heuristic self-score (0-10)."""
        score = 5.0

        word_count = len(text.split())
        if 50 <= word_count <= 2000:
            score += 1.0
        elif word_count < 20:
            score -= 2.0
        elif word_count > 5000:
            score -= 1.0

        if text.count('\n') >= 2:
            score += 0.5

        prompt_lang = self._detect_language(prompt)
        text_lang = self._detect_language(text)
        if prompt_lang == text_lang:
            score += 0.5

        uncertain = heuristic_hallucination_signals(text).get("uncertain_markers", 0)
        score += min(uncertain * 0.2, 1.0)

        signals = heuristic_hallucination_signals(text)
        if signals.get("internal_consistency", {}).get("contradictions_found", 0) > 0:
            score -= 1.5

        if signals.get("overconfident_facts", 0) > 1:
            score -= 1.0

        return max(0.0, min(10.0, score))

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect language of text (simplified)."""
        french_words = {
            "le", "la", "les", "des", "de", "du", "un", "une", "et", "est",
            "dans", "pour", "avec", "sur", "ce", "ceux", "nous", "vous",
            "ils", "elles", "font", "sont", "faire", "dire", "voir",
            "pouvoir", "devoir", "falloir",
        }
        english_words = {
            "the", "a", "an", "is", "are", "was", "were", "in", "for",
            "with", "on", "this", "these", "we", "you", "they", "have",
            "has", "do", "does", "make", "say", "see", "can", "could",
            "would", "should",
        }

        words = set(text.lower().split())
        french_matches = len(words & french_words)
        english_matches = len(words & english_words)

        if french_matches > english_matches:
            return "fr"
        elif english_matches > french_matches:
            return "en"
        return "unknown"

    def handle_result(self, report: QualityReport) -> tuple[bool, str]:
        """Handle quality gate result.

        Returns (should_proceed, message).
        """
        if report.level == QualityLevel.OK:
            return True, "Quality check passed"

        elif report.level == QualityLevel.WARNING:
            if report.recommendation == "regenerate":
                issue_list = ", ".join(i.value for i in report.issues)
                return False, (
                    f"Quality warning: self_score={report.self_score:.1f}, "
                    f"hallucination_rate={report.hallucination_rate:.1%} "
                    f"Issues: {issue_list}"
                )
            else:
                issue_list = ", ".join(i.value for i in report.issues)
                return True, (
                    f"Quality caution: {issue_list}. "
                    f"Review before proceeding."
                )

        else:
            issue_list = ", ".join(i.value for i in report.issues)
            return False, (
                f"Quality FAILED: {issue_list}. "
                f"Automatic fallback recommended."
            )

    def generate_fallback(
        self,
        prompt: str,
        report: Optional[QualityReport] = None,
    ) -> str:
        """Generate a fallback response when quality check fails."""
        if not self.fallback_responses:
            return ""

        if report and report.hallucination_rate > 0.5:
            return (
                "Je ne peux pas répondre à cette question avec certitude. "
                "Les informations fournies contiennent des incohérences ou "
                "des faits non vérifiables."
            )

        if report and any(i == QualityIssue.SELF_SCORE_LOW for i in report.issues):
            return (
                "Je ne suis pas en mesure de fournir une réponse fiable "
                "pour cette question. Je vous recommande de reformuler "
                "ou de fournir plus de contexte."
            )

        return (
            "Je ne peux pas répondre à cette question avec les informations "
            "dont je dispose. Veuillez reformuler votre demande."
        )


class QualityGatePipeline:
    """Pipeline that wraps an LLM call with quality gate + fallback.

    Usage:
        gate = QualityGatePipeline(model="gpt-4o", api_key="...")
        result = gate.call("Explain quantum computing", max_retries=2)
    """

    def __init__(
        self,
        backend,
        config=None,
    ):
        """Initialize pipeline."""
        self.backend = backend
        self.gate = QualityGate(config=config)

    def call(
        self,
        prompt: str,
        system: str = "",
        max_retries: Optional[int] = None,
        check_hallucination: bool = True,
    ) -> dict:
        """Execute LLM call with quality gate + automatic fallback."""
        max_retries = max_retries or self.gate.max_retries
        fallback_used = False
        last_response = None
        last_report = None

        for attempt in range(max_retries + 1):
            response = self.backend.generate(
                system=system or "You are a helpful assistant.",
                prompt=prompt,
            )
            last_response = response

            report = self.gate.evaluate(
                response, prompt, check_hallucination=check_hallucination
            )
            last_report = report

            should_proceed, message = self.gate.handle_result(report)
            if should_proceed:
                logger.info("Quality check passed on attempt %d", attempt + 1)
                return {
                    "response": response,
                    "quality": report,
                    "retry_count": attempt,
                    "fallback_used": False,
                }

            # Last attempt -> fallback
            if attempt == max_retries:
                logger.warning(
                    "Quality check failed after %d attempts. Using fallback.",
                    attempt + 1,
                )
                fallback_used = True
                response = self.gate.generate_fallback(prompt, report)
                return {
                    "response": response,
                    "quality": report,
                    "retry_count": attempt,
                    "fallback_used": True,
                }

            # Regenerate with feedback
            logger.warning(
                "Quality check failed. Attempt %d/%d. Regenerating...",
                attempt + 1,
                max_retries,
            )
            prompt = (
                f"{prompt}\n\n"
                f"Previous attempt had quality issues: {message}\n"
                f"Please provide a more accurate, well-sourced response."
            )

        # Safety net
        return {
            "response": fallback_used and self.gate.generate_fallback(prompt)
            or last_response or "",
            "quality": last_report or QualityReport(
                level=QualityLevel.FAIL,
                self_score=0.0,
                hallucination_rate=1.0,
                citation_count=0,
            ),
            "retry_count": max_retries,
            "fallback_used": True,
        }
