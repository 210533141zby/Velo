import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.evidence_packager import EvidencePackager
from app.services.rag.pipeline_models import (
    DefenseProfile,
    EvidenceAssessment,
    EvidenceRequirement,
    QuestionFocus,
    QuestionFocusType,
    QueryIntent,
    QueryIntentType,
    RetrievedCandidate,
)


def build_assessment(
    doc_id: int,
    *,
    title: str,
    content: str,
    final_score: float,
) -> EvidenceAssessment:
    """构造证据打包测试使用的最小 assessment。"""
    doc = SimpleNamespace(page_content=content, metadata={'source': title, 'doc_id': doc_id})
    return EvidenceAssessment(
        candidate=RetrievedCandidate(
            doc=doc,
            doc_id=doc_id,
            title=title,
            chunk_text=content,
            full_content=content,
        ),
        final_score=final_score,
        usable=True,
    )


class EvidencePackagerTests(unittest.TestCase):
    def test_pack_keeps_reason_paragraph_when_answer_clause_is_late(self) -> None:
        """验证pack keeps原因paragraph when答案条款is late相关行为是否符合预期。"""
        packager = EvidencePackager()
        intent = QueryIntent(
            original_query='为什么额外重视A-17-204的蓝布账册',
            normalized_query='为什么额外重视A-17-204的蓝布账册',
            keyword_query='蓝布 账册 204',
            intent_type=QueryIntentType.FACTOID,
            retrieval_depth=10,
            defense_profile=DefenseProfile.MODERATE,
            evidence_requirement=EvidenceRequirement.MULTI_SPAN,
            wants_short_answer=False,
            needs_judge=False,
            question_focus=QuestionFocus(
                category=QuestionFocusType.REASON,
                cue_text='为什么',
                slot_terms=('A-17-204', '蓝布账册', '重视'),
                prompt_hint='明确回答被询问的原因、依据或动因',
            ),
        )
        content = (
            '第一段介绍档案馆历史。\n\n'
            '档案馆现存纸质资料约十二万六千份，其中最早的一件是清光绪十九年，也就是1893年的《平码渔船借契》。'
            '最受关注的藏品却不是年代最久的，而是一本编号为A-17-204的蓝布账册。'
            '那本账册记录了1949年至1952年间雾潮镇与邻近三处渔村的物资往来，涉及盐、煤油、船钉、稻米和柴油机零件。'
            '研究者之所以重视它，是因为账册里频繁出现一个名字：沈见川。'
        )

        packet = packager.pack(
            '为什么额外重视A-17-204的蓝布账册',
            [build_assessment(38, title='雾潮镇的档案馆', content=content, final_score=0.92)],
            max_units=1,
            char_budget=1200,
            prefer_full_content=True,
            intent=intent,
        )

        self.assertEqual(len(packet.units), 1)
        self.assertIn('A-17-204', packet.units[0].text)
        self.assertIn('沈见川', packet.units[0].text)

    def test_pack_prioritizes_focus_aligned_reason_unit(self) -> None:
        """验证pack prioritizes focus aligned原因unit相关行为是否符合预期。"""
        packager = EvidencePackager()
        intent = QueryIntent(
            original_query='为什么档案馆需要扩建',
            normalized_query='为什么档案馆需要扩建',
            keyword_query='档案馆 需要 扩建',
            intent_type=QueryIntentType.FACTOID,
            retrieval_depth=8,
            defense_profile=DefenseProfile.MODERATE,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
            wants_short_answer=True,
            needs_judge=False,
            question_focus=QuestionFocus(
                category=QuestionFocusType.REASON,
                cue_text='为什么',
                slot_terms=('档案馆', '扩建'),
                prompt_hint='明确回答被询问的原因、依据或动因',
            ),
        )
        assessments = [
            build_assessment(
                1,
                title='扩建通知',
                content='档案馆扩建项目已列入年度计划。\n\n因为现有库房持续饱和，新增档案快速增加，所以需要扩建。',
                final_score=0.9,
            )
        ]

        packet = packager.pack(
            '为什么档案馆需要扩建',
            assessments,
            max_units=1,
            char_budget=800,
            prefer_full_content=True,
            intent=intent,
        )

        self.assertEqual(len(packet.units), 1)
        self.assertIn('因为现有库房持续饱和', packet.units[0].text)

    def test_pack_prefers_primary_source_when_source_scores_are_dominant(self) -> None:
        """验证pack prefers primary来源when来源分数列表are dominant相关行为是否符合预期。"""
        packager = EvidencePackager()
        assessments = [
            build_assessment(
                1,
                title='量子算法综述',
                content=(
                    'Shor算法能够在多项式时间内解决大整数质因数分解问题。\n\n'
                    '它被认为是量子算法中最经典的突破之一。\n\n'
                    '该算法对公钥密码体系具有深远影响。'
                ),
                final_score=0.95,
            ),
            build_assessment(
                2,
                title='密码学背景',
                content='经典密码学常使用大整数分解难题构建安全性假设。',
                final_score=0.63,
            ),
        ]

        packet = packager.pack(
            'Shor算法主要解决什么问题',
            assessments,
            max_units=2,
            char_budget=1200,
            prefer_full_content=True,
        )

        self.assertEqual(len(packet.units), 2)
        self.assertEqual({unit.doc_id for unit in packet.units}, {1})

    def test_pack_selects_diverse_units_from_multiple_sources(self) -> None:
        """验证pack selects diverse units from multiple来源列表相关行为是否符合预期。"""
        packager = EvidencePackager()
        assessments = [
            build_assessment(
                1,
                title='量子算法综述',
                content=(
                    '量子计算是一类利用量子叠加与纠缠的计算模型。\n\n'
                    'Shor算法能够在多项式时间内解决大整数质因数分解问题。\n\n'
                    '它因此成为量子计算早期最具代表性的突破之一。'
                ),
                final_score=0.93,
            ),
            build_assessment(
                2,
                title='密码学背景',
                content=(
                    '经典密码学中，大整数分解被广泛用于公钥体系。\n\n'
                    '一旦分解问题可以高效求解，RSA一类方案将受到直接影响。'
                ),
                final_score=0.88,
            ),
        ]

        packet = packager.pack(
            'Shor算法主要解决什么问题',
            assessments,
            max_units=3,
            char_budget=1600,
            prefer_full_content=True,
        )

        self.assertGreaterEqual(len(packet.units), 2)
        self.assertIn(1, {unit.doc_id for unit in packet.units})
        self.assertTrue(any('质因数分解问题' in unit.text for unit in packet.units))
        self.assertTrue(any(unit.doc_id == 2 for unit in packet.units))

    def test_render_for_synthesis_keeps_unit_identity_without_review_metadata(self) -> None:
        """验证render for synthesis keeps unit identity without审查结果metadata相关行为是否符合预期。"""
        packager = EvidencePackager()
        packet = packager.pack(
            '雾潮镇档案馆建于哪一年',
            [
                build_assessment(
                    38,
                    title='雾潮镇的档案馆',
                    content='档案馆建于1958年，后续又扩建了库房。',
                    final_score=0.91,
                )
            ],
            max_units=2,
            char_budget=1000,
            prefer_full_content=True,
        )

        rendered = packager.render_for_synthesis(packet)

        self.assertIn('证据单元 1', rendered)
        self.assertIn('标题：雾潮镇的档案馆', rendered)
        self.assertIn('\nunit_id: 38:1\n', rendered)
        self.assertIn('\n回答相关度：', rendered)
        self.assertIn('\n内容：\n', rendered)
        self.assertNotIn('doc_id:', rendered)
        self.assertNotIn('来源排序：', rendered)
        self.assertNotIn('\n相关度：', rendered)

    def test_render_for_review_keeps_rank_provenance(self) -> None:
        """验证render for审查结果keeps排序结果provenance相关行为是否符合预期。"""
        packager = EvidencePackager()
        packet = packager.pack(
            '雾潮镇档案馆建于哪一年',
            [
                build_assessment(
                    38,
                    title='雾潮镇的档案馆',
                    content='档案馆建于1958年，后续又扩建了库房。',
                    final_score=0.91,
                )
            ],
            max_units=2,
            char_budget=1000,
            prefer_full_content=True,
        )

        rendered = packager.render_for_review(packet)

        self.assertIn('证据单元 1', rendered)
        self.assertIn('\n来源排序：1\n', rendered)
        self.assertNotIn('doc_id:', rendered)
        self.assertNotIn('相关度：', rendered)


if __name__ == '__main__':
    unittest.main()
