import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.answer_generators import build_sources
from app.services.rag.pipeline_models import EvidenceAssessment, JudgeDecision, RetrievedCandidate
from app.services.rag.rag_service import RagService
from app.services.rag.source_confidence import calibrated_assessment_confidence, calibrated_candidate_confidence


def build_assessment(
    *,
    doc_id: int,
    final_score: float,
    rerank_score: float,
    adaptive_score: float,
    dense_score: float = 0.0,
    bm25_score: float = 0.0,
    coverage_score: float = 0.0,
    identifier_overlap: float = 0.0,
    usable: bool = True,
    direct_evidence: bool = False,
    supports_extractive: bool = False,
    judge_passed: bool = False,
) -> EvidenceAssessment:
    """构造来源置信度测试使用的 assessment。"""
    candidate = RetrievedCandidate(
        doc=SimpleNamespace(page_content='测试内容', metadata={'source': f'文档{doc_id}', 'doc_id': doc_id}),
        doc_id=doc_id,
        title=f'文档{doc_id}',
        rerank_score=rerank_score,
        adaptive_score=adaptive_score,
        dense_score=dense_score,
        bm25_score=bm25_score,
        coverage_score=coverage_score,
        identifier_overlap=identifier_overlap,
    )
    judge = None
    if judge_passed:
        judge = JudgeDecision(
            invoked=True,
            passed=True,
            topic_match=True,
            direct_evidence=direct_evidence,
            answerable=True,
        )
    return EvidenceAssessment(
        candidate=candidate,
        final_score=final_score,
        usable=usable,
        direct_evidence=direct_evidence,
        supports_extractive=supports_extractive,
        judge=judge,
    )


class SourceConfidenceTests(unittest.TestCase):
    def test_calibrated_assessment_confidence_lifts_strong_anchor_case(self) -> None:
        """验证calibrated assessment置信度lifts strong anchor样例相关行为是否符合预期。"""
        strong = build_assessment(
            doc_id=38,
            final_score=0.635,
            rerank_score=1.0,
            adaptive_score=0.065,
            dense_score=0.82,
            bm25_score=0.73,
            coverage_score=0.71,
            identifier_overlap=1.0,
            usable=True,
            direct_evidence=True,
        )

        confidence = calibrated_assessment_confidence(strong, ranked_assessments=[strong])

        self.assertGreater(confidence, 80)
        self.assertGreater(confidence, int(round(strong.final_score * 100)))

    def test_calibrated_assessment_confidence_keeps_weak_case_moderate(self) -> None:
        """验证calibrated assessment置信度keeps weak样例moderate相关行为是否符合预期。"""
        weak = build_assessment(
            doc_id=2,
            final_score=0.44,
            rerank_score=0.41,
            adaptive_score=0.46,
            coverage_score=0.08,
            identifier_overlap=0.0,
            usable=True,
            direct_evidence=False,
            supports_extractive=False,
        )

        confidence = calibrated_assessment_confidence(weak, ranked_assessments=[weak])

        self.assertLess(confidence, 65)

    def test_build_sources_uses_calibrated_confidence(self) -> None:
        """验证build来源列表uses calibrated置信度相关行为是否符合预期。"""
        strong = build_assessment(
            doc_id=38,
            final_score=0.635,
            rerank_score=1.0,
            adaptive_score=0.065,
            dense_score=0.82,
            bm25_score=0.73,
            coverage_score=0.71,
            identifier_overlap=1.0,
            usable=True,
            direct_evidence=True,
        )

        sources = build_sources([strong])

        self.assertEqual(len(sources), 1)
        self.assertGreater(sources[0]['confidence'], 80)

    def test_rag_service_sources_from_assessments_uses_calibration(self) -> None:
        """验证RAG服务来源列表from assessments uses calibration相关行为是否符合预期。"""
        service = RagService(SimpleNamespace())
        strong = build_assessment(
            doc_id=38,
            final_score=0.635,
            rerank_score=1.0,
            adaptive_score=0.065,
            dense_score=0.82,
            bm25_score=0.73,
            coverage_score=0.71,
            identifier_overlap=1.0,
            usable=True,
            direct_evidence=True,
        )

        sources = service._sources_from_assessments([strong])

        self.assertEqual(len(sources), 1)
        self.assertGreater(sources[0]['confidence'], 80)

    def test_calibrated_candidate_confidence_lifts_clear_top_retrieval(self) -> None:
        """验证calibrated candidate置信度lifts clear top retrieval相关行为是否符合预期。"""
        top = RetrievedCandidate(
            doc_id=38,
            title='文档38',
            rerank_score=0.96,
            adaptive_score=0.62,
            dense_score=0.81,
            bm25_score=0.70,
            coverage_score=0.66,
            identifier_overlap=0.90,
        )
        second = RetrievedCandidate(
            doc_id=41,
            title='文档41',
            rerank_score=0.58,
            adaptive_score=0.55,
        )

        confidence = calibrated_candidate_confidence(top, ranked_candidates=[top, second])

        self.assertGreater(confidence, 80)


if __name__ == '__main__':
    unittest.main()
