import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.answer_synthesizer import AnswerSynthesizer
from app.services.rag.pipeline_models import (
    AnswerMode,
    AnswerPlan,
    DefenseProfile,
    EvidenceAssessment,
    EvidenceRequirement,
    QuestionFocus,
    QuestionFocusType,
    QueryIntent,
    QueryIntentType,
    RetrievedCandidate,
)


def build_intent(
    *,
    evidence_requirement: EvidenceRequirement,
    question_focus: QuestionFocus | None = None,
) -> QueryIntent:
    """构造测试用的 QueryIntent 样本。

    测试里会频繁复用同一类问题语境，只在证据需求或焦点类型上做微调，
    因此这里把样板意图对象抽出来统一构造。
    """
    return QueryIntent(
        original_query='Shor算法主要解决什么问题',
        normalized_query='Shor算法主要解决什么问题',
        keyword_query='Shor 算法 解决 问题',
        intent_type=QueryIntentType.FACTOID,
        retrieval_depth=8,
        defense_profile=DefenseProfile.MODERATE,
        evidence_requirement=evidence_requirement,
        wants_short_answer=False,
        needs_judge=False,
        trace_tags=(),
        question_focus=question_focus or QuestionFocus(),
    )


def build_plan(mode: AnswerMode) -> AnswerPlan:
    """构造指定回答模式下的测试计划对象。"""
    return AnswerPlan(
        mode=mode,
        reason='test_mode',
        primary_doc_id=1,
        source_doc_ids=(1,),
        generator_name=f'{mode.value}_generator',
    )


def build_assessment(doc_id: int, title: str, content: str, score: float) -> EvidenceAssessment:
    """构造最小可运行的测试用 EvidenceAssessment。"""
    doc = SimpleNamespace(page_content=content, metadata={'source': title, 'doc_id': doc_id})
    return EvidenceAssessment(
        candidate=RetrievedCandidate(
            doc=doc,
            doc_id=doc_id,
            title=title,
            chunk_text=content,
            full_content=content,
        ),
        final_score=score,
        usable=True,
        direct_evidence=True,
        supports_extractive=False,
    )


class _FakeModel:
    def __init__(self, answer: str = 'Shor算法能够在多项式时间内解决大整数质因数分解问题。') -> None:
        """初始化测试替身模型，并记录收到的提示词。"""
        self.prompts: list[str] = []
        self.answer = answer

    async def ainvoke(self, prompt: str):
        """模拟模型调用，并返回固定结构化回答。"""
        self.prompts.append(prompt)
        return SimpleNamespace(
            content=(
                f'{{"verdict":"answer","answer":"{self.answer}","cited_unit_ids":["1:1"],'
                '"confidence":0.92,"reason":"direct evidence"}'
            )
        )


class AnswerSynthesizerTests(unittest.IsolatedAsyncioTestCase):
    def test_packet_config_expands_budget_for_multi_source_support_cluster(self) -> None:
        """验证packet配置expands budget for multi来源支持信息cluster相关行为是否符合预期。"""
        synthesizer = AnswerSynthesizer(lambda: _FakeModel())
        plan = build_plan(AnswerMode.GENERATIVE)
        intent = build_intent(evidence_requirement=EvidenceRequirement.MULTI_SPAN)
        assessments = [
            build_assessment(1, '量子算法综述', 'Shor算法能够分解大整数。', 0.92),
            build_assessment(2, '密码学背景', 'RSA依赖大整数分解难题。', 0.89),
            build_assessment(3, '量子计算史', 'Shor算法改变了密码学风险评估。', 0.87),
        ]

        config = synthesizer._packet_config(plan, intent, assessments)

        self.assertGreaterEqual(config.max_units, 7)
        self.assertGreaterEqual(config.char_budget, 4700)
        self.assertTrue(config.prefer_full_content)

    async def test_synthesizer_plans_and_answers_in_single_prompt(self) -> None:
        """验证综合器plans and答案列表in single提示词相关行为是否符合预期。"""
        model = _FakeModel()
        synthesizer = AnswerSynthesizer(lambda: model)
        plan = build_plan(AnswerMode.STRUCTURED)
        intent = build_intent(evidence_requirement=EvidenceRequirement.MULTI_SPAN)
        assessments = [
            build_assessment(
                1,
                '量子计算与Shor算法',
                'Shor算法能够在多项式时间内解决大整数质因数分解问题，这被视为量子计算的重要突破。',
                0.91,
            )
        ]

        answer = await synthesizer.synthesize(plan, 'Shor算法主要解决什么问题', intent, assessments)

        self.assertEqual(answer, 'Shor算法能够在多项式时间内解决大整数质因数分解问题。')
        self.assertEqual(len(model.prompts), 1)
        self.assertLess(model.prompts[0].index('用户问题：'), model.prompts[0].index('证据单元：'))
        self.assertIn('你必须先根据证据自行判断答案应是单事实、并列列举、综合说明还是摘要', model.prompts[0])
        self.assertIn('证据单元中的“回答相关度”表示它直接回应问题的可能性', model.prompts[0])
        self.assertIn('用户的核心疑问是：直接概括用户索要的核心事实', model.prompts[0])
        self.assertNotIn('建议答案形态', model.prompts[0])
        self.assertNotIn('"answer_shape"', model.prompts[0])
        self.assertNotIn('"coverage_points"', model.prompts[0])
        self.assertNotIn('"confidence"', model.prompts[0])
        self.assertNotIn('"reason"', model.prompts[0])
        self.assertIn('证据单元 1\nunit_id: 1:1\n', model.prompts[0])
        self.assertIn('\n回答相关度：', model.prompts[0])
        self.assertIn('\n内容：\nShor算法能够在多项式时间内解决大整数质因数分解问题', model.prompts[0])
        self.assertNotIn('doc_id:', model.prompts[0])
        self.assertNotIn('来源排序：', model.prompts[0])
        self.assertNotIn('\n相关度：', model.prompts[0])

    async def test_synthesizer_injects_focus_and_structured_answer_directives(self) -> None:
        """验证综合器injects focus and structured答案directives相关行为是否符合预期。"""
        model = _FakeModel()
        synthesizer = AnswerSynthesizer(lambda: model)
        plan = build_plan(AnswerMode.GENERATIVE)
        intent = build_intent(
            evidence_requirement=EvidenceRequirement.MULTI_SPAN,
            question_focus=QuestionFocus(
                category=QuestionFocusType.CHOICE,
                cue_text='哪些内容',
                slot_terms=('核心', '知识', '角色'),
                prompt_hint='明确回答被询问的对象、选项或并列项，并按问题维度逐条组织',
                expects_multiple_points=True,
            ),
        )
        assessments = [
            build_assessment(1, '知识清单', '第一点是分解问题，第二点是密码学影响。', 0.91),
            build_assessment(2, '角色分工', '研究者需要解释算法原理，实践者需要评估风险。', 0.88),
        ]

        await synthesizer.synthesize(plan, 'Shor算法主要解决什么问题', intent, assessments)

        self.assertIn('用户的核心疑问是：明确回答被询问的对象、选项或并列项，并按问题维度逐条组织', model.prompts[0])
        self.assertIn('需要重点对齐的语义槽位：核心、知识、角色', model.prompts[0])
        self.assertIn('按用户问题中的信息需求组织答案', model.prompts[0])
        self.assertIn('如果有多个并列事实，使用简短列举', model.prompts[0])
        self.assertIn('避免冗长背景和机械拆分', model.prompts[0])

    async def test_synthesizer_aligns_concise_slot_answer_to_query_focus(self) -> None:
        """验证综合器aligns concise槽位答案to问题focus相关行为是否符合预期。"""
        model = _FakeModel(answer='张三')
        synthesizer = AnswerSynthesizer(lambda: model)
        plan = build_plan(AnswerMode.STRUCTURED)
        intent = QueryIntent(
            original_query='谁负责项目验收',
            normalized_query='谁负责项目验收',
            keyword_query='负责 项目 验收',
            intent_type=QueryIntentType.FACTOID,
            retrieval_depth=8,
            defense_profile=DefenseProfile.MODERATE,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
            wants_short_answer=True,
            needs_judge=False,
            question_focus=QuestionFocus(
                category=QuestionFocusType.PERSON,
                cue_text='谁',
                slot_terms=('项目', '验收'),
                prompt_hint='明确回答被询问的人员或主体身份',
            ),
        )
        assessments = [
            build_assessment(1, '项目记录', '张三负责项目验收。', 0.91),
        ]

        answer = await synthesizer.synthesize(plan, '谁负责项目验收', intent, assessments)

        self.assertEqual(answer, '张三负责项目验收')

    async def test_synthesizer_aligns_implicit_attribute_short_answer(self) -> None:
        """验证综合器aligns implicit attribute short答案相关行为是否符合预期。"""
        model = _FakeModel(answer='B站')
        synthesizer = AnswerSynthesizer(lambda: model)
        plan = build_plan(AnswerMode.STRUCTURED)
        intent = QueryIntent(
            original_query='某动画的播放平台',
            normalized_query='某动画的播放平台',
            keyword_query='动画 播放 平台',
            intent_type=QueryIntentType.FACTOID,
            retrieval_depth=8,
            defense_profile=DefenseProfile.MODERATE,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
            wants_short_answer=True,
            needs_judge=False,
            question_focus=QuestionFocus(
                category=QuestionFocusType.ATTRIBUTE,
                cue_text='',
                slot_terms=('播放', '平台'),
                prompt_hint='明确给出问题所求的属性值',
            ),
        )
        assessments = [
            build_assessment(1, '上线记录', '某动画在B站独家上线。', 0.91),
        ]

        answer = await synthesizer.synthesize(plan, '某动画的播放平台', intent, assessments)

        self.assertEqual(answer, '某动画是B站')

    async def test_atomic_synthesis_prompt_discourages_unasked_background(self) -> None:
        """验证atomic synthesis提示词discourages unasked background相关行为是否符合预期。"""
        model = _FakeModel()
        synthesizer = AnswerSynthesizer(lambda: model)
        plan = build_plan(AnswerMode.STRUCTURED)
        intent = build_intent(evidence_requirement=EvidenceRequirement.ATOMIC_SPAN)
        assessments = [
            build_assessment(1, '量子计算与Shor算法', 'Shor算法能够在多项式时间内解决大整数质因数分解问题。', 0.91),
        ]

        await synthesizer.synthesize(plan, 'Shor算法主要解决什么问题', intent, assessments)

        self.assertIn('只回答问题索要的单一事实或属性', model.prompts[0])
        self.assertIn('不要追加时间、地点、背景、评价或来源说明', model.prompts[0])


if __name__ == '__main__':
    unittest.main()
