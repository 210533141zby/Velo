import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.vector_index_service import _prepare_documents_for_indexing, collection_name


class VectorIndexServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_collection_name_changes_with_contextual_variant(self) -> None:
        """验证collection name changes with contextual方案相关行为是否符合预期。"""
        with patch('app.services.rag.vector_index_service.settings.RAG_CONTEXTUAL_EMBED', False):
            base_name = collection_name()
        with patch('app.services.rag.vector_index_service.settings.RAG_CONTEXTUAL_EMBED', True):
            contextual_name = collection_name()

        self.assertNotEqual(base_name, contextual_name)
        self.assertTrue(base_name.endswith('_base'))
        self.assertTrue(contextual_name.endswith('_ctx'))

    async def test_prepare_documents_for_indexing_keeps_raw_text_when_context_disabled(self) -> None:
        """验证prepare文档列表for indexing keeps raw文本when上下文disabled相关行为是否符合预期。"""
        split = SimpleNamespace(page_content='原始段落内容。', metadata={'Header 1': '概述'})

        with patch('app.services.rag.vector_index_service.settings.RAG_CONTEXTUAL_EMBED', False):
            docs = await _prepare_documents_for_indexing(7, '测试文档', '完整文档内容。', [split])

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, '原始段落内容。')
        self.assertEqual(docs[0].metadata['raw_text'], '原始段落内容。')
        self.assertNotIn('contextual_prefix', docs[0].metadata)

    async def test_prepare_documents_for_indexing_adds_context_prefix_and_preserves_raw_text(self) -> None:
        """验证prepare文档列表for indexing adds上下文prefix and preserves raw文本相关行为是否符合预期。"""
        split = SimpleNamespace(page_content='原始段落内容。', metadata={'Header 1': '概述', 'Header 2': '背景'})

        with patch('app.services.rag.vector_index_service.settings.RAG_CONTEXTUAL_EMBED', True), patch(
            'app.services.rag.vector_index_service._generate_contextual_prefix',
            new=AsyncMock(return_value='这段内容位于文档的背景部分，说明核心定义。'),
        ):
            docs = await _prepare_documents_for_indexing(7, '测试文档', '完整文档内容。', [split])

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata['raw_text'], '原始段落内容。')
        self.assertEqual(docs[0].metadata['contextual_prefix'], '这段内容位于文档的背景部分，说明核心定义。')
        self.assertTrue(docs[0].page_content.startswith('这段内容位于文档的背景部分，说明核心定义。'))
        self.assertTrue(docs[0].page_content.endswith('原始段落内容。'))


if __name__ == '__main__':
    unittest.main()
