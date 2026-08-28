"""Tests for quality_gate.quality_gate module."""

from quality_gate.quality_gate import (
    QualityGate,
    QualityGatePipeline,
    QualityIssue,
    QualityLevel,
    QualityReport,
    count_citations,
    heuristic_hallucination_rate,
    heuristic_hallucination_signals,
)


class TestCountCitations:
    def test_bracketed_citations(self):
        text = "Première source [1]. Deuxième source [2]. [3]"
        assert count_citations(text) == 3

    def test_author_year_citations(self):
        text = "(Smith, 2024) et (Martin, 2023)"
        assert count_citations(text) == 2

    def test_no_citations(self):
        text = "Ceci est un texte sans aucune citation."
        assert count_citations(text) == 0

    def test_prefers_bracketed(self):
        text = "Source [1] et (Smith, 2024)"
        assert count_citations(text) == 1  # max(1, 1) = 1


class TestHallucinationSignals:
    def test_fake_citations_detected(self):
        text = "Ref 1: Fake Source — Fake Data"
        signals = heuristic_hallucination_signals(text)
        assert signals["fake_citations"] > 0

    def test_uncertain_markers(self):
        text = "Probablement, il semble que peut-être"
        signals = heuristic_hallucination_signals(text)
        assert signals["uncertain_markers"] >= 2

    def test_overconfident_facts(self):
        text = "C'est prouvé, c'est certain, la vérité c'est que"
        signals = heuristic_hallucination_signals(text)
        assert signals["overconfident_facts"] >= 2

    def test_internal_consistency(self):
        text = "Le chat est un animal. Le chat n'est pas un animal."
        signals = heuristic_hallucination_signals(text)
        assert signals["internal_consistency"]["contradictions_found"] > 0

    def test_clean_text(self):
        text = "Je ne suis pas sûr de cette information. Il semble que plusieurs sources soient contradictoires."
        signals = heuristic_hallucination_signals(text)
        assert signals["fake_citations"] == 0
        assert signals["overconfident_facts"] == 0


class TestHeuristicHallucinationRate:
    def test_clean_text_low_rate(self):
        text = "Je ne suis pas certain de cette information. Les sources semblent contradictoires."
        rate = heuristic_hallucination_rate(text, "")
        assert rate < 0.1

    def test_fake_citation_high_rate(self):
        text = "Ref 1: Fake Source — Fake Data. Ref 2: Another Fake — More Fake."
        rate = heuristic_hallucination_rate(text, "")
        assert rate > 0.1


class TestQualityGate:
    def test_clean_text_pass(self):
        gate = QualityGate()
        text = "Voici une réponse bien structurée avec des sources [1] et [2]."
        report = gate.evaluate(text, "Question: Qu'est-ce que la lune?")
        assert report.level == QualityLevel.OK

    def test_no_citations_warn(self):
        gate = QualityGate(config=dict(min_citations=2))
        text = "La lune est un satellite naturel de la Terre (Wikipedia)."
        report = gate.evaluate(text, "Question: Qu'est-ce que la lune?")
        assert report.level in (QualityLevel.WARNING, QualityLevel.OK)

    def test_fake_citation_fail(self):
        gate = QualityGate()
        text = "Ref 1: Fake Source — Fake Data"
        report = gate.evaluate(text, "Question: Qu'est-ce que la lune?")
        assert report.level == QualityLevel.FAIL

    def test_short_text_fail(self):
        gate = QualityGate()
        text = "Ok"
        report = gate.evaluate(text, "Question: Explique la physique quantique.")
        assert report.level == QualityLevel.FAIL

    def test_contradiction_fail(self):
        gate = QualityGate()
        text = "Le chat est un animal. Le chat n'est pas un animal."
        report = gate.evaluate(text, "Question: Qu'est-ce qu'un chat?")
        assert report.level == QualityLevel.FAIL


class TestQualityReport:
    def test_to_dict(self):
        report = QualityReport(
            level=QualityLevel.OK,
            self_score=8.0,
            hallucination_rate=0.05,
            citation_count=3,
        )
        d = report.to_dict()
        assert d["level"] == "ok"
        assert d["self_score"] == 8.0
        assert d["hallucination_rate"] == 0.05
        assert d["citation_count"] == 3
        assert d["issues"] == []


class TestQualityGateHandleResult:
    def test_ok_result(self):
        gate = QualityGate()
        report = QualityReport(
            level=QualityLevel.OK,
            self_score=8.0,
            hallucination_rate=0.05,
            citation_count=3,
        )
        should_proceed, message = gate.handle_result(report)
        assert should_proceed is True
        assert "passed" in message

    def test_fail_result(self):
        gate = QualityGate()
        report = QualityReport(
            level=QualityLevel.FAIL,
            self_score=2.0,
            hallucination_rate=0.8,
            citation_count=0,
            issues=[QualityIssue.HALLUCINATION_RATE],
        )
        should_proceed, message = gate.handle_result(report)
        assert should_proceed is False
        assert "FAILED" in message


class TestFallback:
    def test_hallucination_fallback(self):
        gate = QualityGate()
        report = QualityReport(
            level=QualityLevel.FAIL,
            self_score=2.0,
            hallucination_rate=0.8,
            citation_count=0,
        )
        fallback = gate.generate_fallback("test prompt", report)
        assert "certitude" in fallback or "incertitude" in fallback

    def test_default_fallback(self):
        gate = QualityGate()
        fallback = gate.generate_fallback("test prompt", None)
        assert "repondre" in fallback or "formuler" in fallback


class TestDetectLanguage:
    def test_french(self):
        from quality_gate.quality_gate import QualityGate
        assert QualityGate._detect_language("le chat est un animal") == "fr"

    def test_english(self):
        from quality_gate.quality_gate import QualityGate
        assert QualityGate._detect_language("the cat is an animal") == "en"


# --- LLMBackend mock tests ---

class TestQualityGatePipeline:
    """Tests using a mock LLM backend."""

    def test_pass_on_first_attempt(self):
        """When LLM generates good text, return immediately."""
        from quality_gate.backend import LLMBackend

        class MockBackend:
            def generate(self, system, prompt, **kwargs):
                return "Réponse bien structurée. Source [1]. Source [2]."

        pipeline = QualityGatePipeline(MockBackend())
        result = pipeline.call("Question?")
        assert result["fallback_used"] is False
        assert result["retry_count"] == 0
        assert result["quality"].level == QualityLevel.OK

    def test_fallback_on_quality_fail(self):
        """When LLM generates bad text, use fallback."""
        call_count = [0]

        class MockBackend:
            def generate(self, system, prompt, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return "Ref 1: Fake Source — Fake Data"
                return "Deuxième tentative"

        pipeline = QualityGatePipeline(MockBackend(), config=dict(max_retries=1))
        result = pipeline.call("Question?")
        assert result["fallback_used"] is True
        assert result["quality"].level == QualityLevel.FAIL
        assert "repondre" in result["response"].lower() or "reformuler" in result["response"].lower()

    def test_pass_on_second_attempt(self):
        """When LLM fixes bad text on second try, return it."""
        call_count = [0]

        class MockBackend:
            def generate(self, system, prompt, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return "Ref 1: Fake Source — Fake Data"
                return "Réponse bien structurée avec des sources [1] et [2]."

        pipeline = QualityGatePipeline(MockBackend(), config=dict(max_retries=2))
        result = pipeline.call("Question?")
        assert result["fallback_used"] is False
        assert result["retry_count"] == 1  # Second attempt
        assert result["quality"].level == QualityLevel.OK
