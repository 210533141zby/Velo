import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.rerank_service import build_rerank_input


class RerankServiceTests(unittest.TestCase):
    def test_build_rerank_input_prefers_raw_text_over_contextual_prefix(self) -> None:
        """验证build重排结果input prefers raw文本over contextual prefix相关行为是否符合预期。"""
        doc = SimpleNamespace(
            page_content='[context prefix]\\n\\n正文',
            metadata={'source': 't', 'raw_text': '正文'},
        )

        rendered = build_rerank_input(doc)

        self.assertEqual(rendered, '标题：t\n内容：正文')
        self.assertNotIn('context prefix', rendered)


if __name__ == '__main__':
    unittest.main()
