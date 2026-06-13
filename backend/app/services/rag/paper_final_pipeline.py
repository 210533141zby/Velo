from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from app.services.rag.answer_synthesizer import align_short_answer_to_focus
from app.services.rag.pipeline_models import EvidenceAssessment, EvidenceRequirement, QueryIntent, QuestionFocusType
from app.services.rag.prompt_templates import build_no_context_answer
from app.services.rag.rerank_service import normalize_lookup_text, tokenize_text
from app.services.rag.text_utils import compact_structured_text, compact_text, split_paragraphs, split_text_segments

NO_CONTEXT_ANSWER = build_no_context_answer()
TIME_SPAN_PATTERN = re.compile(r'(?:\d{2,4}年|\d{1,2}月|\d{1,2}日|\d{1,2}[点时分]|上午|下午|凌晨|当日|当天)')
NUMBER_SPAN_PATTERN = re.compile(
    r'(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万两]+)(?:[%％]|个|人|名|项|次|家|位|届|天|年|月|小时|分钟|公里|米|亿元?|万元?|倍|级)?'
)
REASON_MARKERS = ('因为', '由于', '之所以', '缘于', '原因在于', '原因是', '因此', '所以')
METHOD_MARKERS = ('通过', '采用', '利用', '借助', '依靠', '方式', '方法', '步骤')
_DETAIL_RE = re.compile(
    r"(?:约|近|逾|超|超过|至少|不足|将近)?\d+(?:\.\d+)?"
    r"(?:万|亿|千|百)?(?:%|％|年|月|日|元|美元|亿元|亿美元|人|名|个|件|项|次|家|位|届|公里|米|吨|级|号)?|"
    r"[A-Z]{2,}[A-Za-z0-9:/-]*"
)
SHORT_CLAUSE_KEEP_PATTERN = re.compile(
    r"(?:意义|作用|原因|缘由|方法|方式|定义|是什么|特点|独特之处|区别|影响|时间|日期|地点|内容|有哪些|哪几点)"
)


@dataclass(frozen=True)
class PaperFinalUnit:
    doc_id: int | None
    title: str
    text: str
    source_score: float
    unit_score: float
    overall_coverage: float
    clause_coverages: tuple[float, ...]
    aligned_clause: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class PaperFinalResult:
    answer: str
    selected_assessments: tuple[EvidenceAssessment, ...]
    used_complex_path: bool


def _coverage_ratio(left: set[str], right: set[str]) -> float:
    """计算候选内容对查询词的覆盖率。"""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _jaccard(left: set[str], right: set[str]) -> float:
    """计算两个集合的 Jaccard 相似度。"""
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _signature(value: str) -> str:
    """生成文本签名。"""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


def _clean_query(value: str, limit: int = 240) -> str:
    """清洗查询。"""
    return compact_text(value, limit)


def extract_query_aspect_clauses(query: str, *, max_clauses: int = 4) -> list[str]:
    """把复杂问题拆成若干有顺序的分项子句。"""
    cleaned = _clean_query(query)
    if not cleaned:
        return []

    primary_parts = [
        segment.strip(" ，,；;。！？? ")
        for segment in re.split(
            r"(?:以及|并且|同时|并(?:说明|指出|给出|列举|列出|比较|分析|描述|判断)|[，,]\s*且| and | as well as )",
            cleaned,
        )
        if segment.strip(" ，,；;。！？? ")
    ]
    clauses: list[str] = []
    for part in primary_parts:
        secondary_parts = [
            segment.strip(" ，,；;。！？? ")
            for segment in re.split(r"[，,；;]", part)
            if segment.strip(" ，,；;。！？? ")
        ]
        clauses.extend(secondary_parts if secondary_parts else [part])

    normalized: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        clause = re.sub(r"^(?:说明|解释|指出|概括|分析|补充)\s*", "", clause).strip()
        if len(clause) < 3 and not SHORT_CLAUSE_KEEP_PATTERN.search(clause):
            continue
        signature = _signature(clause)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        normalized.append(clause)
        if len(normalized) >= max_clauses:
            break
    return normalized


def estimate_parallel_requirement_count(query: str, *, max_count: int = 4) -> int:
    """估计问题中并行信息需求的大致数量。

    论文主链路在决定是否走复杂路径时，需要知道用户是不是一次问了多个并列要求。
    这里通过连接词和显式并列提示做一个轻量估计，而不是依赖额外模型判断。
    """
    cleaned = _clean_query(query)
    if not cleaned:
        return 1
    count = 1
    count += cleaned.count("以及")
    count += cleaned.count("并且")
    count += cleaned.count("同时")
    count += cleaned.count("并说明")
    count += cleaned.count("并指出")
    count += cleaned.count("并给出")
    count += cleaned.count("并列出")
    count += cleaned.count("并比较")
    count += cleaned.count("并分析")
    count += cleaned.count("，且")
    count += cleaned.count(",且")
    if "分别" in cleaned and re.search(r"[和与及、]", cleaned):
        count = max(count, 2)
    return max(1, min(max_count, count))


def infer_query_task_mode(query: str) -> str:
    """推断查询任务模式。"""
    cleaned = _clean_query(query)
    normalized = normalize_lookup_text(cleaned)
    edit_markers = ("修改", "纠正", "改写", "润色", "续写", "改成", "改为")
    if any(marker in normalized for marker in edit_markers):
        return "edit"
    if any(marker in cleaned for marker in ("概括", "总结", "归纳", "概述", "核心内容", "主要内容", "简述")):
        return "summary"
    if any(marker in cleaned for marker in ("谁", "何时", "什么时候", "哪天", "哪一年", "多少", "哪些", "哪个", "哪里", "什么")):
        return "factoid"
    return "default"


def should_use_complex_paper_path(
    query: str,
    intent: QueryIntent,
    assessments: Sequence[EvidenceAssessment],
) -> bool:
    """判断当前问题是否应进入论文主链路的复杂作答路径。

    触发条件主要来自三类信号：问题本身是多分项结构、焦点天然要求多点回答，
    或者证据需求已明显超出原子片段层级。该函数的目的是把复杂路径的入口条件
    集中在一处，方便实验和线上部署保持一致。
    """
    clauses = extract_query_aspect_clauses(query, max_clauses=4)
    unique_doc_count = len(
        {
            assessment.candidate.doc_id if assessment.candidate.doc_id is not None else assessment.candidate.title
            for assessment in assessments
        }
    )
    if len(clauses) >= 2:
        return True
    if intent.question_focus.expects_multiple_points:
        return True
    if (
        intent.evidence_requirement is EvidenceRequirement.MULTI_SPAN
        and intent.question_focus.has_explicit_cue
        and intent.question_focus.category in {QuestionFocusType.REASON, QuestionFocusType.METHOD}
    ):
        return True
    if estimate_parallel_requirement_count(query) >= 2:
        return True
    return unique_doc_count >= 2 and intent.evidence_requirement is not EvidenceRequirement.ATOMIC_SPAN


def _source_content(assessment: EvidenceAssessment) -> str:
    """提取来源文档的正文内容。"""
    candidate = assessment.candidate
    if candidate.full_content.strip():
        return compact_structured_text(candidate.full_content, 3600)
    if candidate.chunk_text.strip():
        return compact_structured_text(candidate.chunk_text, 2400)
    page_content = str(getattr(candidate.doc, "page_content", "") or "")
    return compact_structured_text(page_content, 2400)


def _candidate_windows(content: str) -> list[str]:
    """把正文切成论文主链路内部使用的候选窗口。

    这套窗口比普通抽取式回答更偏保守，既保留段落结构，也控制窗口数量，
    主要服务于后面的分项覆盖分析和细节信号估计。
    """
    paragraphs = split_paragraphs(content)
    if len(paragraphs) >= 2:
        windows: list[str] = []
        for paragraph in paragraphs[:8]:
            compacted = compact_text(paragraph, 720)
            if compacted:
                windows.append(compacted)
        for index in range(min(len(paragraphs) - 1, 4)):
            combined = compact_structured_text('\n\n'.join(paragraphs[index : index + 2]), 720)
            if combined:
                windows.append(combined)
        return windows

    sentences = split_text_segments(content)
    if not sentences:
        compacted = compact_text(content, 720)
        return [compacted] if compacted else []
    if len(sentences) <= 2:
        compacted = compact_structured_text("\n".join(sentences), 720)
        return [compacted] if compacted else []

    windows: list[str] = []
    for index in range(len(sentences)):
        window = compact_structured_text("\n".join(sentences[index : index + 2]), 720)
        if window:
            windows.append(window)
        if len(windows) >= 4:
            break
    return windows


def _detail_signal(text: str) -> float:
    """估计文本中的细节信号强度。"""
    return min(len(_DETAIL_RE.findall(text)), 3) / 3.0


def _reorder_assessments(
    query: str,
    assessments: Sequence[EvidenceAssessment],
) -> list[EvidenceAssessment]:
    """按论文主链路规则重新排序评估结果。"""
    clauses = extract_query_aspect_clauses(query, max_clauses=4)
    if len(clauses) < 2:
        return sorted(assessments, key=lambda item: float(item.final_score), reverse=True)

    query_tokens = set(tokenize_text(query))
    clause_token_sets = [set(tokenize_text(clause)) for clause in clauses if tokenize_text(clause)]
    if len(clause_token_sets) < 2:
        return sorted(assessments, key=lambda item: float(item.final_score), reverse=True)

    prepared: list[tuple[EvidenceAssessment, float]] = []
    for assessment in assessments:
        windows = _candidate_windows(_source_content(assessment))
        if not windows:
            continue
        best_coverage = 0.0
        best_aspect_ratio = 0.0
        best_detail = 0.0
        for window in windows:
            tokens = set(tokenize_text(window))
            if not tokens:
                continue
            best_coverage = max(best_coverage, _coverage_ratio(query_tokens, tokens))
            covered_aspects = sum(1 for clause_tokens in clause_token_sets if _coverage_ratio(clause_tokens, tokens) >= 0.22)
            best_aspect_ratio = max(best_aspect_ratio, covered_aspects / max(len(clause_token_sets), 1))
            best_detail = max(best_detail, _detail_signal(window))
        conservative_score = (
            0.70 * float(assessment.final_score)
            + 0.16 * best_aspect_ratio
            + 0.10 * best_coverage
            + 0.04 * best_detail
        )
        prepared.append((assessment, conservative_score))

    if not prepared:
        return sorted(assessments, key=lambda item: float(item.final_score), reverse=True)
    prepared.sort(key=lambda item: (item[1], float(item[0].candidate.rerank_score), float(item[0].final_score)), reverse=True)
    return [assessment for assessment, _ in prepared]


def _build_units(
    query: str,
    assessments: Sequence[EvidenceAssessment],
    *,
    source_limit: int,
) -> list[PaperFinalUnit]:
    """把评估结果整理成证据单元。"""
    clauses = extract_query_aspect_clauses(query, max_clauses=4)
    query_tokens = set(tokenize_text(query))
    clause_token_sets = [set(tokenize_text(clause)) for clause in clauses if tokenize_text(clause)]
    units: list[PaperFinalUnit] = []

    for source_rank, assessment in enumerate(assessments[: max(source_limit, 1)], start=1):
        windows = _candidate_windows(_source_content(assessment))
        for unit_index, window in enumerate(windows, start=1):
            tokens = set(tokenize_text(window))
            if not tokens:
                continue
            overall_coverage = _coverage_ratio(query_tokens, tokens)
            if clause_token_sets:
                clause_coverages = tuple(_coverage_ratio(clause_tokens, tokens) for clause_tokens in clause_token_sets)
                best_clause_index = max(range(len(clause_coverages)), key=lambda idx: clause_coverages[idx])
                aligned_clause = clauses[best_clause_index]
                best_clause_score = clause_coverages[best_clause_index]
            else:
                clause_coverages = ()
                aligned_clause = ""
                best_clause_score = overall_coverage
            position_bonus = max(0.0, 1.0 - (unit_index - 1) * 0.18)
            unit_score = (
                0.56 * float(assessment.final_score)
                + 0.18 * overall_coverage
                + 0.12 * best_clause_score
                + 0.08 * _detail_signal(window)
                + 0.06 * position_bonus
            )
            units.append(
                PaperFinalUnit(
                    doc_id=assessment.candidate.doc_id,
                    title=assessment.candidate.title,
                    text=window,
                    source_score=float(assessment.final_score),
                    unit_score=unit_score,
                    overall_coverage=overall_coverage,
                    clause_coverages=clause_coverages,
                    aligned_clause=aligned_clause,
                    tokens=frozenset(tokens),
                )
            )
    units.sort(key=lambda item: (item.unit_score, item.source_score, item.overall_coverage), reverse=True)
    return units


def _select_covering_units(
    query: str,
    units: Sequence[PaperFinalUnit],
    *,
    final_limit: int,
) -> list[PaperFinalUnit]:
    """选择能覆盖问题分项的证据单元。"""
    clauses = extract_query_aspect_clauses(query, max_clauses=4)
    if final_limit <= 0 or not units:
        return []
    if len(clauses) < 2:
        selected: list[PaperFinalUnit] = []
        seen_doc_keys: set[object] = set()
        for unit in units:
            doc_key = unit.doc_id if unit.doc_id is not None else unit.title
            if doc_key in seen_doc_keys:
                continue
            selected.append(unit)
            seen_doc_keys.add(doc_key)
            if len(selected) >= final_limit:
                break
        return selected or list(units[:final_limit])

    clause_count = len(clauses)
    selected: list[PaperFinalUnit] = [units[0]]
    selected_tokens = [set(units[0].tokens)]
    seen_doc_keys = {units[0].doc_id if units[0].doc_id is not None else units[0].title}
    covered_clauses = [0.0] * clause_count
    for index, coverage in enumerate(units[0].clause_coverages):
        covered_clauses[index] = max(covered_clauses[index], float(coverage))

    while len(selected) < final_limit:
        best_unit: PaperFinalUnit | None = None
        best_score = float("-inf")
        unresolved_clause_exists = any(value < 0.18 for value in covered_clauses)
        for unit in units:
            if unit in selected:
                continue
            clause_coverages = list(unit.clause_coverages) or [unit.overall_coverage] * clause_count
            residual_gains = [max(0.0, coverage - covered_clauses[idx]) for idx, coverage in enumerate(clause_coverages)]
            residual_gain = sum(residual_gains) / max(len(residual_gains), 1)
            uncovered_hit_ratio = sum(
                1 for idx, coverage in enumerate(clause_coverages) if covered_clauses[idx] < 0.18 and coverage >= 0.18
            ) / max(clause_count, 1)
            strong_fill_ratio = sum(
                1 for idx, coverage in enumerate(clause_coverages) if covered_clauses[idx] < 0.30 and coverage >= 0.30
            ) / max(clause_count, 1)
            redundancy = max((_jaccard(set(unit.tokens), existing) for existing in selected_tokens), default=0.0)
            doc_key = unit.doc_id if unit.doc_id is not None else unit.title
            if doc_key not in seen_doc_keys and uncovered_hit_ratio > 0.0:
                new_doc_bonus = 1.0
            elif doc_key not in seen_doc_keys:
                new_doc_bonus = 0.35
            else:
                new_doc_bonus = 0.0
            same_doc_penalty = 1.0 if doc_key in seen_doc_keys and uncovered_hit_ratio <= 0.0 else 0.0
            score = (
                0.34 * residual_gain
                + 0.24 * uncovered_hit_ratio
                + 0.14 * strong_fill_ratio
                + 0.10 * unit.overall_coverage
                + 0.10 * unit.source_score
                + 0.08 * _detail_signal(unit.text)
                + 0.08 * new_doc_bonus
                - 0.16 * redundancy
                - 0.08 * same_doc_penalty
            )
            if unresolved_clause_exists and uncovered_hit_ratio <= 0.0:
                score -= 0.20
            if score > best_score:
                best_score = score
                best_unit = unit
        if best_unit is None:
            break
        selected.append(best_unit)
        selected_tokens.append(set(best_unit.tokens))
        seen_doc_keys.add(best_unit.doc_id if best_unit.doc_id is not None else best_unit.title)
        clause_coverages = list(best_unit.clause_coverages) or [best_unit.overall_coverage] * clause_count
        for index, coverage in enumerate(clause_coverages):
            covered_clauses[index] = max(covered_clauses[index], float(coverage))

    return selected


def _build_task_aligned_prompt(query: str, units: Sequence[PaperFinalUnit]) -> str:
    """构建按任务形态对齐的作答提示词。"""
    blocks: list[str] = []
    for index, unit in enumerate(units, start=1):
        block_lines = [f"[{index}] {unit.title}"]
        if unit.aligned_clause:
            block_lines.append(f"对应要求：{unit.aligned_clause}")
        block_lines.append(unit.text)
        blocks.append("\n".join(block_lines))
    context = "\n\n".join(blocks)

    task_mode = infer_query_task_mode(query)
    aspect_count = estimate_parallel_requirement_count(query)
    task_lines: list[str] = []
    if task_mode == "factoid":
        task_lines.extend(
            [
                "- 这是属性定位任务。回答必须写成一整句自包含陈述，明确写出对象、属性和值，不要只输出裸日期、数字、名称或地点短语。",
                "- 若某个属性无法确定，直接在句中说明无法从给定证据确定。",
            ]
        )
    elif task_mode == "summary":
        task_lines.append("- 这是概括任务。优先用 1 到 2 句覆盖主体、动作、结果或处理机制，避免无关背景扩写。")

    if aspect_count >= 2 and task_mode != "edit":
        aspect_clauses = extract_query_aspect_clauses(query, max_clauses=aspect_count)
        if aspect_clauses:
            task_lines.append("- 问题包含多个并列要求。回答必须按原顺序覆盖下列要求，不得只回答其中一部分：")
            for index, clause in enumerate(aspect_clauses, start=1):
                task_lines.append(f"  {index}. {clause}")
            task_lines.append(
                f"- 优先按上述顺序用 {len(aspect_clauses)} 个紧凑分句或句子覆盖；如果某一项证据不足，明确指出该项无法从给定证据确定。"
            )
            task_lines.append("- 若问题要求列举内容、措施、原因、产品、企业、条件或数量，不要只做笼统概括；应保留关键条目和总量信息。")

    task_block = "\n".join(task_lines)
    if task_block:
        task_block += "\n"

    return (
        "你是一个严格的证据问答助手。请仅根据下列证据单元回答用户问题。\n"
        "你需要先在心里判断最小充分回答形态，再输出最终答案；不要展示思考过程。\n"
        "- 回答必须直接对齐用户问题的表达框架，优先沿用问题中的主语、谓语和限定条件。\n"
        "- 回答应当自包含：即使单独读这一句，也能看出是在回答哪个对象、事件或结论。\n"
        "- 若证据中已经给出可直接填入问题的时间、数字、专名、范围或结论，优先复用原词，不做无必要改写。\n"
        "- 涉及时间、日期、数量、地点或范围时，保留证据中的完整限定形式，不要省略单位、起止边界等必要信息。\n"
        f"{task_block}"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n"
        "- 不要编造证据中没有的事实，不要输出 JSON、编号、项目符号、来源说明，避免“根据资料/根据证据”这类套话。\n\n"
        f"证据单元：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def _normalize_answer(query: str, intent: QueryIntent, raw_answer: str) -> str:
    """规范化答案。"""
    cleaned = compact_text(str(raw_answer or "").replace("**", ""), 960).strip()
    cleaned = re.sub(r"^(?:回答[:：]|答案[:：])\s*", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    if cleaned == NO_CONTEXT_ANSWER:
        return cleaned
    if len(cleaned) <= 64:
        cleaned = align_short_answer_to_focus(query, intent, cleaned)
    return cleaned.strip()


def paper_final_answer_is_acceptable(query: str, intent: QueryIntent, answer: str) -> bool:
    """判断论文主链路生成的答案是否可接受。"""
    del query
    cleaned = compact_text(str(answer or "").replace("**", ""), 960).strip()
    if not cleaned:
        return False
    if cleaned == NO_CONTEXT_ANSWER or NO_CONTEXT_ANSWER in cleaned:
        return False

    focus = intent.question_focus.category
    if focus is QuestionFocusType.TIME:
        return bool(TIME_SPAN_PATTERN.search(cleaned) or NUMBER_SPAN_PATTERN.search(cleaned))
    if focus is QuestionFocusType.QUANTITY:
        return bool(NUMBER_SPAN_PATTERN.search(cleaned))
    if focus is QuestionFocusType.REASON:
        if any(marker in cleaned for marker in REASON_MARKERS):
            return True
        return not any(marker in cleaned for marker in ('原因不明', '无法确定', '证据不足', '信息不足'))
    if focus is QuestionFocusType.METHOD:
        return any(marker in cleaned for marker in METHOD_MARKERS)
    return True


async def generate_paper_final_answer(
    model_getter: Callable,
    query: str,
    intent: QueryIntent,
    assessments: Sequence[EvidenceAssessment],
    *,
    final_source_count: int,
    complex_source_count: int,
) -> PaperFinalResult:
    """运行论文主链路并生成最终答案。"""
    ranked_assessments = _reorder_assessments(query, assessments)
    candidate_units = _build_units(query, ranked_assessments, source_limit=complex_source_count)
    final_units = _select_covering_units(query, candidate_units, final_limit=final_source_count)
    if not final_units:
        return PaperFinalResult(answer="", selected_assessments=(), used_complex_path=True)

    selected_keys: list[object] = []
    for unit in final_units:
        selected_keys.append(unit.doc_id if unit.doc_id is not None else unit.title)
    selected_assessments: list[EvidenceAssessment] = []
    seen_keys: set[object] = set()
    for key in selected_keys:
        if key in seen_keys:
            continue
        for assessment in ranked_assessments:
            doc_key = assessment.candidate.doc_id if assessment.candidate.doc_id is not None else assessment.candidate.title
            if doc_key == key:
                selected_assessments.append(assessment)
                seen_keys.add(key)
                break

    prompt = _build_task_aligned_prompt(query, final_units)
    fallback_answer = ""
    candidate_texts: list[str] = []
    seen_signatures: set[str] = set()
    for unit in final_units:
        text = compact_structured_text(unit.text, 360)
        signature = _signature(text)
        if not text or not signature or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidate_texts.append(text)
    if candidate_texts:
        fallback_answer = "；".join(
            candidate_texts[: max(len(extract_query_aspect_clauses(query, max_clauses=4)), 1)]
        )

    try:
        response = await asyncio.wait_for(model_getter().ainvoke(prompt), timeout=18.0)
        answer = _normalize_answer(query, intent, getattr(response, "content", ""))
    except Exception:
        # 在线演示阶段优先保证“有依据地回答出来”，
        # 因此模型调用异常时退回到证据单元拼接答案，而不是直接整条链路失败。
        answer = _normalize_answer(query, intent, fallback_answer)
    if not answer or answer == NO_CONTEXT_ANSWER:
        answer = _normalize_answer(query, intent, fallback_answer)
    return PaperFinalResult(
        answer=answer,
        selected_assessments=tuple(selected_assessments),
        used_complex_path=True,
    )
