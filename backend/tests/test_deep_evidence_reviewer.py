import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.deep_evidence_reviewer import deep_review_and_answer
from app.services.rag.pipeline_models import RetrievedCandidate


def build_candidate(
    doc_id: int,
    *,
    title: str,
    content: str,
    adaptive_score: float,
    rerank_score: float,
    coverage_score: float,
) -> RetrievedCandidate:
    """构造深度复核测试使用的候选文档对象。"""
    doc = SimpleNamespace(page_content=content, metadata={'source': title, 'doc_id': doc_id})
    return RetrievedCandidate(
        doc=doc,
        doc_id=doc_id,
        title=title,
        adaptive_score=adaptive_score,
        rerank_score=rerank_score,
        coverage_score=coverage_score,
        chunk_text=content,
        full_content=content,
    )


class _FakeModel:
    def __init__(self) -> None:
        """初始化深度复核测试替身模型。"""
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str):
        """记录提示词并返回固定的复核结果。"""
        self.prompts.append(prompt)
        return SimpleNamespace(
            content=(
                'review complete\n```json\n'
                '{"verdict":"answer","answer":"Shor算法能够分解大整数。","used_ranks":[]}\n'
                '```'
            )
        )


class DeepEvidenceReviewerTests(unittest.IsolatedAsyncioTestCase):
    async def test_deep_review_uses_shared_evidence_packet_context(self) -> None:
        """验证deep审查结果uses shared证据packet上下文相关行为是否符合预期。"""
        model = _FakeModel()
        candidates = [
            build_candidate(
                1,
                title='量子计算与Shor算法',
                content='Shor算法能够在多项式时间内解决大整数质因数分解问题。',
                adaptive_score=0.84,
                rerank_score=0.91,
                coverage_score=0.78,
            ),
            build_candidate(
                2,
                title='密码学背景',
                content='一旦分解问题可以高效求解，RSA一类方案将受到直接影响。',
                adaptive_score=0.79,
                rerank_score=0.86,
                coverage_score=0.73,
            ),
        ]

        with patch('app.services.rag.deep_evidence_reviewer.get_chat_model', return_value=model):
            result = await deep_review_and_answer('Shor算法主要解决什么问题', candidates)

        self.assertEqual(set(result.keys()), {'verdict', 'answer', 'used_ranks'})
        self.assertEqual(result['verdict'], 'answer')
        self.assertEqual(result['used_ranks'], (1, 2))
        self.assertEqual(len(model.prompts), 1)
        self.assertLess(model.prompts[0].index('用户问题：'), model.prompts[0].index('证据单元：'))
        self.assertIn('证据单元 1', model.prompts[0])
        self.assertIn('来源排序：1', model.prompts[0])
        self.assertNotIn('doc_id:', model.prompts[0])
        self.assertNotIn('相关度：', model.prompts[0])
        self.assertNotIn('候选块 1', model.prompts[0])
        self.assertNotIn('"answer_shape"', model.prompts[0])
        self.assertNotIn('"coverage_points"', model.prompts[0])
        self.assertNotIn('"conflicts"', model.prompts[0])


if __name__ == '__main__':
    unittest.main()
