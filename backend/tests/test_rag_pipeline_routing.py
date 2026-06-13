import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.answer_mode_router import AnswerModeRouter
from app.services.rag.pipeline_models import (
    AnswerMode,
    DefenseProfile,
    EvidenceAssessment,
    EvidenceRequirement,
    QuestionFocus,
    QuestionFocusType,
    QueryIntent,
    QueryIntentType,
    RetrievedCandidate,
)
from app.services.rag.query_intent_builder import QueryIntentBuilder


def build_intent(
    intent_type: QueryIntentType,
    *,
    evidence_requirement: EvidenceRequirement,
    wants_short_answer: bool = True,
    question_focus: QuestionFocus | None = None,
) -> QueryIntent:
    """构造路由测试使用的 QueryIntent。"""
    return QueryIntent(
        original_query='测试问题',
        normalized_query='测试问题',
        keyword_query='测试 问题',
        intent_type=intent_type,
        retrieval_depth=8,
        defense_profile=DefenseProfile.MODERATE,
        evidence_requirement=evidence_requirement,
        wants_short_answer=wants_short_answer,
        needs_judge=False,
        trace_tags=(),
        question_focus=question_focus or QuestionFocus(),
    )


def build_assessment(
    doc_id: int,
    *,
    final_score: float = 0.9,
    usable: bool = True,
    direct_evidence: bool = True,
    supports_extractive: bool = True,
) -> EvidenceAssessment:
    """构造路由测试使用的 EvidenceAssessment。"""
    return EvidenceAssessment(
        candidate=RetrievedCandidate(doc_id=doc_id, title=f'文档{doc_id}'),
        final_score=final_score,
        usable=usable,
        direct_evidence=direct_evidence,
        supports_extractive=supports_extractive,
    )


class QueryIntentBuilderTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_relation_intent_for_relation_query(self) -> None:
        """验证builds relation意图for relation问题相关行为是否符合预期。"""
        intent = await QueryIntentBuilder.build('韩国与新加坡的关系')

        self.assertEqual(intent.intent_type, QueryIntentType.RELATION)
        self.assertEqual(intent.defense_profile, DefenseProfile.MODERATE)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.MULTI_SPAN)
        self.assertFalse(intent.needs_judge)

    async def test_builds_multi_span_intent_for_descriptive_fragment(self) -> None:
        """验证builds multi片段意图for descriptive fragment相关行为是否符合预期。"""
        intent = await QueryIntentBuilder.build('雾潮镇的历史')

        self.assertEqual(intent.intent_type, QueryIntentType.FACTOID)
        self.assertEqual(intent.defense_profile, DefenseProfile.MODERATE)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.MULTI_SPAN)
        self.assertFalse(intent.wants_short_answer)

    async def test_builds_lookup_intent_for_short_entity_query(self) -> None:
        """验证builds lookup意图for short entity问题相关行为是否符合预期。"""
        intent = await QueryIntentBuilder.build('沈阳')

        self.assertEqual(intent.intent_type, QueryIntentType.LOOKUP)
        self.assertEqual(intent.defense_profile, DefenseProfile.STRICT)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.ATOMIC_SPAN)
        self.assertIn('short_query', intent.trace_tags)

    async def test_builds_factoid_intent_for_specific_fact_question(self) -> None:
        """验证builds factoid意图for specific事实question相关行为是否符合预期。"""
        intent = await QueryIntentBuilder.build('档案馆建于哪一年')

        self.assertEqual(intent.intent_type, QueryIntentType.FACTOID)
        self.assertEqual(intent.defense_profile, DefenseProfile.MODERATE)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.ATOMIC_SPAN)
        self.assertIn('档案馆', intent.keyword_query)
        self.assertEqual(intent.question_focus.category, QuestionFocusType.TIME)
        self.assertTrue(intent.question_focus.has_explicit_cue)

    async def test_builds_multi_span_intent_for_attribute_style_fragment(self) -> None:
        """验证builds multi片段意图for attribute style fragment相关行为是否符合预期。"""
        intent = await QueryIntentBuilder.build('新加坡的档案馆')

        self.assertEqual(intent.intent_type, QueryIntentType.FACTOID)
        self.assertEqual(intent.defense_profile, DefenseProfile.MODERATE)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.MULTI_SPAN)
        self.assertFalse(intent.needs_judge)
        self.assertNotIn('attribute_style_query', intent.trace_tags)
        self.assertEqual(intent.question_focus.category, QuestionFocusType.ATTRIBUTE)

    async def test_identifier_anchored_reason_query_keeps_multi_span_but_not_multi_point_focus(self) -> None:
        """验证identifier anchored原因问题keeps multi片段but not multi point focus相关行为是否符合预期。"""
        intent = await QueryIntentBuilder.build('为什么研究者重视编号为A-17-204的蓝布账册')

        self.assertEqual(intent.intent_type, QueryIntentType.FACTOID)
        self.assertEqual(intent.defense_profile, DefenseProfile.MODERATE)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.MULTI_SPAN)
        self.assertFalse(intent.needs_judge)
        self.assertNotIn('为什么', intent.keyword_query)
        self.assertEqual(intent.question_focus.category, QuestionFocusType.REASON)
        self.assertFalse(intent.question_focus.expects_multiple_points)


class AnswerModeRouterTests(unittest.TestCase):
    def test_routes_to_no_context_when_no_usable_docs(self) -> None:
        """验证routes to no上下文when no usable文档列表相关行为是否符合预期。"""
        intent = build_intent(
            QueryIntentType.FACTOID,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
        )

        plan = AnswerModeRouter.route(intent, [])

        self.assertEqual(plan.mode, AnswerMode.NO_CONTEXT)
        self.assertEqual(plan.reason, 'no_usable_evidence')

    def test_routes_factoid_with_direct_evidence_to_extractive(self) -> None:
        """验证routes factoid with direct证据to extractive相关行为是否符合预期。"""
        intent = build_intent(
            QueryIntentType.FACTOID,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
        )

        plan = AnswerModeRouter.route(intent, [build_assessment(38, final_score=0.94)])

        self.assertEqual(plan.mode, AnswerMode.EXTRACTIVE)
        self.assertEqual(plan.generator_name, 'extractive_generator')
        self.assertEqual(plan.primary_doc_id, 38)

    def test_routes_short_answer_without_precise_span_to_structured(self) -> None:
        """验证routes short答案without precise片段to structured相关行为是否符合预期。"""
        intent = build_intent(
            QueryIntentType.RELATION,
            evidence_requirement=EvidenceRequirement.MULTI_SPAN,
        )

        plan = AnswerModeRouter.route(
            intent,
            [build_assessment(38, final_score=0.81, direct_evidence=False, supports_extractive=False)],
        )

        self.assertEqual(plan.mode, AnswerMode.STRUCTURED)
        self.assertEqual(plan.generator_name, 'structured_generator')
        self.assertEqual(plan.reason, 'evidence_grounded_synthesis')

    def test_routes_overview_to_generative(self) -> None:
        """验证routes overview to generative相关行为是否符合预期。"""
        intent = build_intent(
            QueryIntentType.OVERVIEW,
            evidence_requirement=EvidenceRequirement.FULL_DOCUMENT,
            wants_short_answer=False,
        )

        plan = AnswerModeRouter.route(intent, [build_assessment(38, final_score=0.91)])

        self.assertEqual(plan.mode, AnswerMode.GENERATIVE)
        self.assertEqual(plan.reason, 'full_document_requirement')

    def test_routes_multi_document_relation_to_generative(self) -> None:
        """验证routes multi文档relation to generative相关行为是否符合预期。"""
        intent = build_intent(
            QueryIntentType.RELATION,
            evidence_requirement=EvidenceRequirement.MULTI_SPAN,
        )

        plan = AnswerModeRouter.route(
            intent,
            [
                build_assessment(38, final_score=0.88, direct_evidence=True, supports_extractive=False),
                build_assessment(41, final_score=0.76, direct_evidence=True, supports_extractive=False),
            ],
        )

        self.assertEqual(plan.mode, AnswerMode.GENERATIVE)
        self.assertEqual(plan.reason, 'multi_document_synthesis')

    def test_routes_single_document_multi_span_factoid_to_structured_without_planner(self) -> None:
        """验证routes single文档multi片段factoid to structured without planner相关行为是否符合预期。"""
        intent = build_intent(
            QueryIntentType.FACTOID,
            evidence_requirement=EvidenceRequirement.MULTI_SPAN,
            wants_short_answer=False,
        )

        plan = AnswerModeRouter.route(
            intent,
            [build_assessment(38, final_score=0.87, direct_evidence=True, supports_extractive=False)],
        )

        self.assertEqual(plan.mode, AnswerMode.STRUCTURED)
        self.assertEqual(plan.reason, 'evidence_grounded_synthesis')

    def test_routes_ambiguous_atomic_multi_source_case_to_generative(self) -> None:
        """验证routes ambiguous atomic multi来源样例to generative相关行为是否符合预期。"""
        intent = build_intent(
            QueryIntentType.FACTOID,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
        )

        plan = AnswerModeRouter.route(
            intent,
            [
                build_assessment(38, final_score=0.88, direct_evidence=True, supports_extractive=False),
                build_assessment(41, final_score=0.84, direct_evidence=True, supports_extractive=False),
            ],
        )

        self.assertEqual(plan.mode, AnswerMode.GENERATIVE)
        self.assertEqual(plan.reason, 'multi_source_atomic_resolution')
        self.assertIn('clear_margin', plan.trace_data)
        self.assertIn('support_margin', plan.trace_data)
        self.assertIn('dominance_ratio', plan.trace_data)
        self.assertFalse(plan.trace_data['primary_is_clear'])

    def test_routes_explicit_focus_atomic_query_to_extractive_even_without_span_flag(self) -> None:
        """验证routes explicit focus atomic问题to extractive even without片段flag相关行为是否符合预期。"""
        intent = build_intent(
            QueryIntentType.FACTOID,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
            question_focus=QuestionFocus(
                category=QuestionFocusType.TIME,
                cue_text='哪一年',
                slot_terms=('档案馆', '建于'),
                prompt_hint='明确回答被询问的时间、时段或日期',
            ),
        )

        plan = AnswerModeRouter.route(
            intent,
            [build_assessment(38, final_score=0.92, direct_evidence=True, supports_extractive=False)],
        )

        self.assertEqual(plan.mode, AnswerMode.EXTRACTIVE)
        self.assertTrue(plan.trace_data['question_focus_explicit'])

    def test_routes_reason_question_to_structured_instead_of_extractive(self) -> None:
        """验证routes原因question to structured instead of extractive相关行为是否符合预期。"""
        intent = build_intent(
            QueryIntentType.FACTOID,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
            question_focus=QuestionFocus(
                category=QuestionFocusType.REASON,
                cue_text='为什么',
                slot_terms=('蓝布账册', '重视'),
                prompt_hint='明确回答被询问的原因、依据或动因',
            ),
        )

        plan = AnswerModeRouter.route(
            intent,
            [build_assessment(38, final_score=0.92, direct_evidence=True, supports_extractive=True)],
        )

        self.assertEqual(plan.mode, AnswerMode.STRUCTURED)
        self.assertEqual(plan.reason, 'atomic_synthesis_required')
