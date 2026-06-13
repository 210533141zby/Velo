from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Callable, Sequence

from app.logger import logger
from app.services.rag.evidence_packager import EvidencePackager
from app.services.rag.llm_json_utils import extract_json_dict, strip_markdown_code_fence
from app.services.rag.pipeline_models import (
    AnswerMode,
    AnswerPlan,
    EvidenceAssessment,
    EvidenceRequirement,
    QueryIntent,
    QuestionFocusType,
)
from app.services.rag.prompt_templates import build_answer_synthesis_prompt
from app.services.rag.score_distribution import summarize_scores
from app.services.rag.text_utils import compact_text

DEFAULT_ANSWER_LIMIT = 960
SLOT_FILLING_ANSWER_LIMIT = 18


@dataclass(frozen=True)
class PacketConfig:
    max_units: int
    char_budget: int
    prefer_full_content: bool


@dataclass(frozen=True)
class SynthesisDirectives:
    question_focus: str
    focus_terms: str
    response_format: str


def _normalize_answer(value: str, limit: int) -> str:
    """规范化答案。"""
    cleaned = compact_text(strip_markdown_code_fence(value), limit).strip()
    cleaned = re.sub(r'^(?:回答[:：]|答案[:：])\s*', '', cleaned)
    return cleaned.strip()


def _fallback_answer_from_packet_texts(texts: Sequence[str], limit: int) -> str:
    """在生成阶段超时或失败时，把高相关证据直接整理成保底答案。

    这个兜底不再依赖模型，只取排在前面的证据单元做轻量压缩与拼接，
    目标是在答辩演示场景里优先返回“有据可查的内容”，而不是长时间挂起。
    """
    snippets: list[str] = []
    seen: set[str] = set()
    for raw_text in texts:
        text = compact_text(raw_text, 240).strip('；;，, ')
        if not text or text in seen:
            continue
        seen.add(text)
        snippets.append(text)
        if len(snippets) >= 3:
            break
    return compact_text('；'.join(snippets), limit).strip()


def align_short_answer_to_focus(query: str, intent: QueryIntent, answer: str) -> str:
    """在结构安全的前提下，把短答案重新嵌回用户问题框架。

    这一步的适用范围刻意收得很窄，只处理非常短、且明显对应单一槽位的答案。
    这样做的目的是改善回答表述与问题的贴合度，同时避免引入额外实体、
    领域规则或面向特定基准的硬编码。
    """
    focus = intent.question_focus
    cleaned = str(answer or '').strip()
    if (
        not cleaned
        or len(cleaned) > SLOT_FILLING_ANSWER_LIMIT
        or any(symbol in cleaned for symbol in '，,；;。！？!?')
    ):
        return answer
    if focus.category not in {
        QuestionFocusType.TIME,
        QuestionFocusType.LOCATION,
        QuestionFocusType.QUANTITY,
        QuestionFocusType.PERSON,
        QuestionFocusType.CHOICE,
        QuestionFocusType.DEFINITION,
        QuestionFocusType.ATTRIBUTE,
    }:
        return answer

    if focus.has_explicit_cue and focus.cue_text in query:
        replacement = cleaned
        if focus.category is QuestionFocusType.LOCATION and focus.cue_text.startswith('在') and not replacement.startswith('在'):
            replacement = f'在{replacement}'
        rewritten = query.replace(focus.cue_text, replacement, 1)
        return rewritten.strip('，,。.!！？?：:；; ')

    if not focus.has_explicit_cue and focus.category is QuestionFocusType.ATTRIBUTE and focus.slot_terms:
        subject = _implicit_slot_subject(query, focus.slot_terms)
        if subject:
            return f'{subject}是{cleaned}'

    return answer


def _implicit_slot_subject(query: str, slot_terms: tuple[str, ...]) -> str:
    """补足隐式属性问句中的稳定主语框架。"""
    cleaned_query = str(query or '').strip('，,。.!！？?：:；; ')
    if not cleaned_query:
        return ''
    subject = cleaned_query
    terms = sorted((str(item).strip() for item in slot_terms if str(item).strip()), key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        for term in terms:
            if term and subject.endswith(term) and len(subject) > len(term):
                subject = subject[: -len(term)].strip('的 ，,。.!！？?：:；; ')
                changed = True
                break
    if subject and subject != cleaned_query:
        return subject
    return cleaned_query


class AnswerSynthesizer:
    """在单次模型调用中同时完成规划与作答的证据约束综合器。"""

    def __init__(self, model_getter: Callable) -> None:
        """初始化当前对象需要的状态和依赖。"""
        self.model_getter = model_getter
        self.packager = EvidencePackager()

    async def synthesize(
        self,
        plan: AnswerPlan,
        query: str,
        intent: QueryIntent,
        assessments: Sequence[EvidenceAssessment],
    ) -> str:
        """执行答案综合。"""
        if not assessments:
            return ''

        config = self._packet_config(plan, intent, assessments)
        packet = self.packager.pack(
            query,
            assessments,
            max_units=config.max_units,
            char_budget=config.char_budget,
            prefer_full_content=config.prefer_full_content,
            intent=intent,
        )
        if not packet.units:
            return ''

        directives = self._build_directives(plan, intent, packet)
        prompt = build_answer_synthesis_prompt(
            query,
            self.packager.render_for_synthesis(packet),
            question_focus=directives.question_focus,
            focus_terms=directives.focus_terms,
            response_format=directives.response_format,
        )
        answer_limit = self._answer_limit(intent)
        fallback_answer = self._fallback_answer(packet, answer_limit)

        try:
            response = await asyncio.wait_for(self.model_getter().ainvoke(prompt), timeout=22.0)
        except asyncio.TimeoutError:
            logger.warning(
                '答案综合超时，回退到证据式答案',
                extra={'extra_data': {'event': 'rag_answer_synthesis_timeout', 'query': query}},
            )
            return fallback_answer
        except Exception:
            logger.exception(
                '答案综合失败，回退到证据式答案',
                extra={'extra_data': {'event': 'rag_answer_synthesis_failed', 'query': query}},
            )
            return fallback_answer

        raw_content = getattr(response, 'content', '')
        payload = extract_json_dict(raw_content)

        if payload:
            verdict = str(payload.get('verdict') or '').strip().lower()
            answer = _normalize_answer(str(payload.get('answer') or ''), answer_limit)
            if verdict == 'answer' and answer:
                return align_short_answer_to_focus(intent.normalized_query, intent, answer)
            if verdict == 'no_context':
                return fallback_answer

        answer = _normalize_answer(str(raw_content or ''), answer_limit)
        if not answer:
            return fallback_answer
        return align_short_answer_to_focus(intent.normalized_query, intent, answer)

    def _packet_config(
        self,
        plan: AnswerPlan,
        intent: QueryIntent,
        assessments: Sequence[EvidenceAssessment],
    ) -> PacketConfig:
        """生成当前综合阶段使用的证据打包配置。"""
        unique_doc_count = len(
            {
                assessment.candidate.doc_id if assessment.candidate.doc_id is not None else assessment.candidate.title
                for assessment in assessments
            }
        )
        distribution = summarize_scores(
            [float(assessment.final_score) for assessment in assessments],
            fallback_clear_margin=0.12,
            min_clear_margin=0.06,
            max_clear_margin=0.16,
            min_support_margin=0.04,
            max_support_margin=0.12,
        )

        if intent.evidence_requirement is EvidenceRequirement.FULL_DOCUMENT or plan.mode is AnswerMode.GENERATIVE:
            base_units = 6
            base_chars = 4200
        elif intent.evidence_requirement is EvidenceRequirement.MULTI_SPAN:
            base_units = 5
            base_chars = 3400
        else:
            base_units = 3
            base_chars = 2200

        multi_source_cluster = unique_doc_count > 1 and distribution.support_cluster_size > 1
        dominant_primary = unique_doc_count > 1 and distribution.dominance_ratio >= 1.0
        support_bonus = max(0, min(distribution.support_cluster_size - 1, 2))
        ambiguity_bonus = 0 if dominant_primary else 1
        max_units = min(8, base_units + support_bonus + (1 if multi_source_cluster else ambiguity_bonus))
        char_budget = min(5600, base_chars + support_bonus * 500 + (500 if multi_source_cluster else ambiguity_bonus * 300))
        prefer_full_content = (
            plan.mode is AnswerMode.GENERATIVE
            or intent.evidence_requirement is not EvidenceRequirement.ATOMIC_SPAN
            or multi_source_cluster
            or not dominant_primary
        )
        return PacketConfig(
            max_units=max_units,
            char_budget=char_budget,
            prefer_full_content=prefer_full_content,
        )

    def _build_directives(
        self,
        plan: AnswerPlan,
        intent: QueryIntent,
        packet,
    ) -> SynthesisDirectives:
        """构建综合阶段的额外约束指令。"""
        unique_doc_count = len(
            {
                unit.doc_id if unit.doc_id is not None else unit.title
                for unit in packet.units
            }
        )
        should_structure = (
            intent.question_focus.expects_multiple_points
            or intent.evidence_requirement is EvidenceRequirement.MULTI_SPAN
            or unique_doc_count > 1 and len(packet.units) > 1
            or plan.mode is AnswerMode.GENERATIVE and len(packet.units) >= 3
        )
        if (
            intent.question_focus.category is QuestionFocusType.REASON
            and not intent.question_focus.expects_multiple_points
        ):
            response_format = (
                '第一句直接回答问题中的原因，不要先复述背景；'
                '如果需要补充，只保留支撑该原因的最少事实。'
            )
        elif (
            intent.question_focus.category is QuestionFocusType.METHOD
            and not intent.question_focus.expects_multiple_points
        ):
            response_format = (
                '第一句直接给出问题所问的方法或做法，不要先铺垫背景；'
                '如果需要补充，只保留落实该方法所必需的信息。'
            )
        elif should_structure:
            response_format = (
                '按用户问题中的信息需求组织答案；如果有多个并列事实，使用简短列举；'
                '如果存在不同回答维度，用自然分句或短段区分，避免冗长背景和机械拆分。'
            )
        elif intent.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN:
            response_format = (
                '只回答问题索要的单一事实或属性；第一句复用用户问题的主体与关系框架完成填槽，'
                '不要追加时间、地点、背景、评价或来源说明，除非这些信息本身就是问题所问。'
            )
        elif intent.question_focus.prefers_slot_filling:
            response_format = '先用第一句把问题中的关键槽位直接填满，再补充最少必要的证据化说明。'
        else:
            response_format = '保持第一句直答，只补充支撑该答案所必需的最少信息。'
        focus_terms = '、'.join(intent.question_focus.slot_terms)
        return SynthesisDirectives(
            question_focus=intent.question_focus.prompt_hint,
            focus_terms=focus_terms,
            response_format=response_format,
        )

    def _answer_limit(self, intent: QueryIntent) -> int:
        """计算当前答案的长度上限。"""
        if intent.evidence_requirement is EvidenceRequirement.FULL_DOCUMENT:
            return 1200
        if intent.evidence_requirement is EvidenceRequirement.MULTI_SPAN:
            return 800
        return DEFAULT_ANSWER_LIMIT

    def _fallback_answer(self, packet, answer_limit: int) -> str:
        """基于已打包的证据单元生成无需模型参与的保底答案。"""
        return _fallback_answer_from_packet_texts([unit.text for unit in packet.units], answer_limit)
