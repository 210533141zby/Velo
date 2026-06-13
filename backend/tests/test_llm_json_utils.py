import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.llm_json_utils import extract_json_dict, strip_markdown_code_fence


class LlmJsonUtilsTests(unittest.TestCase):
    def test_strip_markdown_code_fence_removes_outer_wrapper(self) -> None:
        """验证strip Markdown code fence removes outer wrapper相关行为是否符合预期。"""
        value = '```json\n{"verdict":"answer","answer":"42"}\n```'

        self.assertEqual(strip_markdown_code_fence(value), '{"verdict":"answer","answer":"42"}')

    def test_extract_json_dict_recovers_embedded_object(self) -> None:
        """验证extract JSON 结果dict recovers embedded object相关行为是否符合预期。"""
        value = '分析完成。```json\n{"verdict":"answer","answer":"Shor算法可分解大整数"}\n```请直接使用。'

        self.assertEqual(
            extract_json_dict(value),
            {'verdict': 'answer', 'answer': 'Shor算法可分解大整数'},
        )


if __name__ == '__main__':
    unittest.main()
