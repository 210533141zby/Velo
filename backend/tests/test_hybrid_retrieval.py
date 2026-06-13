import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.hybrid_search import (
    build_hybrid_candidates,
    compute_query_profile,
    ensure_hybrid_index,
    extract_identifiers,
    has_identifier,
)


class FakeDoc:
    def __init__(self, title: str, doc_id: int, page_content: str, metadata: dict | None = None) -> None:
        """初始化当前对象需要的状态和依赖。"""
        self.metadata = {'source': title, 'doc_id': doc_id, **(metadata or {})}
        self.page_content = page_content


class HybridRetrievalTests(unittest.TestCase):
    def test_identifier_extraction_language_neutral(self) -> None:
        """验证identifier extraction language neutral相关行为是否符合预期。"""
        cases = [
            ('CSAPP 第3章', True, {'csapp'}),
            ('Qwen2.5 模型', True, {'qwen2.5'}),
            ('gpt-4 和 gpt-4o 区别', True, {'gpt-4', 'gpt-4o'}),
            ('API-v1 接口', True, {'api-v1'}),
            ('RFC-9110 内容', True, {'rfc-9110'}),
            ('Wi-Fi 7', True, {'wi-fi'}),
            ('MySQL8 新特性', True, {'mysql8'}),
            ('OpenAI o1 模型', True, {'openai'}),
            ('Node.js 20', True, {'node'}),
            ('BGE-M3 嵌入模型', True, {'bge-m3'}),
            ('安倍晋三', False, set()),
            ('2022', False, set()),
            ('2024年高考', False, set()),
            ('北京大学', False, set()),
            ('数据库事务', False, set()),
            ('第三章', False, set()),
            ('三次工业革命', False, set()),
            ('哪一年发布', False, set()),
            ('量子计算', False, set()),
            ('春运时间', False, set()),
        ]

        for query, expected_flag, expected_terms in cases:
            with self.subTest(query=query):
                identifiers = extract_identifiers(query)
                self.assertEqual(has_identifier(query), expected_flag)
                if not expected_flag:
                    self.assertFalse(identifiers)
                for term in expected_terms:
                    self.assertIn(term, identifiers)

    def test_query_profile_increases_lexical_weight_for_identifier_query(self) -> None:
        """验证问题profile increases lexical weight for identifier问题相关行为是否符合预期。"""
        indexed = ensure_hybrid_index(
            [
                SimpleNamespace(id=1, title='CSAPP 第3章', content='讲解链接、装载与共享库', updated_at=None),
                SimpleNamespace(id=2, title='数据库事务', content='介绍隔离级别与并发控制', updated_at=None),
            ]
        )

        profile = compute_query_profile('CSAPP 第3章讲了什么', indexed.idf_lookup)

        self.assertGreater(profile['lexical_weight'], 0.5)
        self.assertLess(profile['dense_weight'], 0.5)

    def test_query_profile_does_not_treat_year_only_query_as_identifier(self) -> None:
        """验证问题profile does not treat year only问题as identifier相关行为是否符合预期。"""
        indexed = ensure_hybrid_index(
            [
                SimpleNamespace(id=1, title='2024年高考', content='介绍考试安排与分数发布时间。', updated_at=None),
                SimpleNamespace(id=2, title='数据库事务', content='介绍隔离级别与并发控制', updated_at=None),
            ]
        )

        profile = compute_query_profile('2024年高考', indexed.idf_lookup)

        self.assertEqual(profile['has_identifier'], 0.0)
        self.assertLess(profile['lexical_weight'], 0.61)

    def test_hybrid_candidates_can_recover_bm25_only_document(self) -> None:
        """验证混合检索candidates can recover BM25 only文档相关行为是否符合预期。"""
        indexed = ensure_hybrid_index(
            [
                SimpleNamespace(id=1, title='CSAPP 第3章', content='这一章主要讲链接、装载与共享库。', updated_at=None),
                SimpleNamespace(id=2, title='Python 教程', content='主要介绍基础语法。', updated_at=None),
                SimpleNamespace(id=3, title='CSAPP 第4章', content='这一章主要讲处理器体系结构。', updated_at=None),
            ]
        )
        vector_matches = [
            (FakeDoc('Python 教程', 2, '主要介绍基础语法。'), 0.91),
        ]

        candidates = build_hybrid_candidates(
            'CSAPP 第3章讲了什么',
            vector_matches,
            indexed,
            vector_limit=10,
            bm25_limit=10,
            candidate_limit=5,
        )

        candidate_ids = [item[0].metadata['doc_id'] for item in candidates]
        self.assertIn(1, candidate_ids)
        recovered = next(doc for doc, _score in candidates if doc.metadata['doc_id'] == 1)
        self.assertEqual(recovered.metadata['candidate_source'], 'hybrid')
        self.assertGreater(recovered.metadata['bm25_score'], 0.0)
        self.assertGreater(recovered.metadata['adaptive_score'], 0.0)


if __name__ == '__main__':
    unittest.main()
