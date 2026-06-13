import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.rag import _build_effective_query
from app.schemas import ChatMessage
from app.services.rag.query_intent_builder import build_structure_profile, infer_question_focus


class RagFollowupContextTests(unittest.TestCase):
    def test_build_effective_query_keeps_full_question_for_regular_turn(self) -> None:
        """验证build effective问题keeps full question for regular turn相关行为是否符合预期。"""
        messages = [
            ChatMessage(role='user', content='雾潮镇档案馆的正式名称是什么'),
        ]

        effective_query = _build_effective_query(messages)

        self.assertEqual(effective_query, '雾潮镇档案馆的正式名称是什么')

    def test_build_effective_query_expands_followup_with_previous_turn(self) -> None:
        """验证build effective问题expands追问with previous turn相关行为是否符合预期。"""
        messages = [
            ChatMessage(role='user', content='潮雾镇的老榕树种植于什么时候'),
            ChatMessage(role='assistant', content='其中一株老榕树种植于1962年。'),
            ChatMessage(role='user', content='另一株呢'),
        ]

        effective_query = _build_effective_query(messages)

        self.assertIn('潮雾镇的老榕树种植于什么时候', effective_query)
        self.assertIn('上一轮回答：其中一株老榕树种植于1962年', effective_query)
        self.assertIn('补充问题：另一株呢', effective_query)

    def test_time_focus_can_keep_multi_point_signal(self) -> None:
        """验证time focus can keep multi point signal相关行为是否符合预期。"""
        query = '潮雾镇的老榕树种植于什么时候'

        focus = infer_question_focus(query, build_structure_profile(query))

        self.assertEqual(focus.category.value, 'time')
        self.assertTrue(focus.expects_multiple_points)


if __name__ == '__main__':
    unittest.main()
