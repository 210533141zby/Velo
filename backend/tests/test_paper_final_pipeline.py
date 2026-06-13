import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.paper_final_pipeline import (
    _build_units,
    paper_final_answer_is_acceptable,
    should_use_complex_paper_path,
)
from app.services.rag.pipeline_models import (
    EvidenceAssessment,
    QueryIntent,
    QueryIntentType,
    QuestionFocus,
    QuestionFocusType,
    RetrievedCandidate,
    DefenseProfile,
    EvidenceRequirement,
)


def build_intent() -> QueryIntent:
    """构造论文主链路测试使用的问题意图。"""
    return QueryIntent(
        original_query='为什么额外重视A-17-204的蓝布账册',
        normalized_query='204的蓝布账册',
        keyword_query='蓝布 账册 204',
        intent_type=QueryIntentType.FACTOID,
        retrieval_depth=10,
        defense_profile=DefenseProfile.MODERATE,
        evidence_requirement=EvidenceRequirement.MULTI_SPAN,
        wants_short_answer=False,
        needs_judge=False,
        trace_tags=('question_focus_reason',),
        question_focus=QuestionFocus(
            category=QuestionFocusType.REASON,
            cue_text='为什么',
            slot_terms=('A-17-204', '蓝布账册', '重视'),
            prompt_hint='明确回答被询问的原因、依据或动因',
            expects_multiple_points=False,
        ),
    )


def build_assessment() -> EvidenceAssessment:
    """构造论文主链路测试使用的证据评估对象。"""
    content = (
        '第一段介绍档案馆历史。\n\n'
        '第二段仍然是背景信息。\n\n'
        '档案馆现存纸质资料约十二万六千份，其中最受关注的藏品却不是年代最久的，而是一本编号为A-17-204的蓝布账册。'
        '研究者之所以重视它，是因为账册里频繁出现一个名字：沈见川。'
    )
    return EvidenceAssessment(
        candidate=RetrievedCandidate(
            doc_id=38,
            title='雾潮镇的档案馆',
            chunk_text=content,
            full_content=content,
        ),
        final_score=0.92,
        usable=True,
        direct_evidence=True,
        supports_extractive=False,
    )


class PaperFinalPipelineTests(unittest.TestCase):
    def test_reason_focus_query_uses_paper_final_even_without_multi_point_shape(self) -> None:
        """验证原因focus问题uses paper final even without multi point shape相关行为是否符合预期。"""
        intent = build_intent()

        self.assertTrue(
            should_use_complex_paper_path(
                intent.original_query,
                intent,
                [build_assessment()],
            )
        )

    def test_build_units_can_reach_late_reason_paragraph(self) -> None:
        """验证build units can reach late原因paragraph相关行为是否符合预期。"""
        units = _build_units(
            '为什么额外重视A-17-204的蓝布账册',
            [build_assessment()],
            source_limit=4,
        )

        self.assertTrue(units)
        self.assertIn('A-17-204', units[0].text)
        self.assertIn('沈见川', units[0].text)
        self.assertGreater(units[0].overall_coverage, 0.0)

    def test_rejects_paper_final_answer_that_embeds_no_context_template(self) -> None:
        """验证rejects paper final答案that embeds no上下文template相关行为是否符合预期。"""
        intent = build_intent()
        answer = '为什么额外重视A-17-204的蓝布账册原因不明，根据当前检索到的知识库内容，没有找到足够相关的参考资料，因此我暂时无法给出可靠回答。'

        self.assertFalse(paper_final_answer_is_acceptable(intent.original_query, intent, answer))


if __name__ == '__main__':
    unittest.main()
