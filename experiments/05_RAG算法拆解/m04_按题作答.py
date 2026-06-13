"""论文中“按题作答”的独立实现。

这一层只负责回答一件事：证据已经排好、选好之后，
怎样让回答模型按题目要求组织输出，而不是泛泛概括。
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Sequence

from m00_公共基础 import (
    DEFAULT_LLM_MODEL,
    NO_CONTEXT_ANSWER,
    OLLAMA_GENERATE_API,
    apply_model_prompt_controls,
    clean_text,
    normalize_text,
)

_COMPLETION_CACHE: dict[tuple[str, int, str], str] = {}


def build_simple_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """生成最简直接作答提示词。"""
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    return (
        "你是一个问答助手。请只依据下列证据单元回答用户问题。\n"
        "- 直接给出自然语言答案，不要输出 JSON、编号、项目符号、来源说明或思考过程。\n"
        "- 如果证据里已经出现关键专名、时间、数字、范围或结论，尽量沿用原词，不要无必要泛化改写。\n"
        "- 若问题涉及多个事实点，用紧凑连贯的方式覆盖完整，不要扩写无关背景。\n"
        "- 只有在证据确实无法确定答案时，才明确说明无法从给定材料确定；不要因为证据分散在多段就直接拒答。\n\n"
        f"证据单元：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_aligned_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """生成与问题表达框架强对齐的提示词。"""
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    return (
        "你是一个严格的证据问答助手。请仅根据下列证据单元回答用户问题。\n"
        "你需要先在心里判断最小充分回答形态，再输出最终答案；不要展示思考过程。\n"
        "- 回答必须直接对齐用户问题的表达框架，优先沿用问题中的主语、谓语和限定条件。\n"
        "- 回答应当自包含：即使单独读这一句，也能看出是在回答哪个对象、事件或结论。\n"
        "- 若证据中已经给出可直接填入问题的时间、数字、专名或结论，优先复用原词，不做无必要改写。\n"
        "- 涉及时间、日期、数量、地点或范围时，保留证据中的完整限定形式，不要省略单位、时区、起止边界等必要信息。\n"
        "- 如果一个短事实就能回答，优先写成一整句自包含陈述，不补充无关背景。\n"
        "- 如果问题需要多个事实点，按问题涉及的维度依次作答，每句只保留一个清晰信息点。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n"
        "- 不要编造证据中没有的事实，不要输出 JSON、编号、项目符号、来源说明，避免“根据资料/根据证据”这类套话。\n\n"
        f"证据单元：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def infer_query_task_mode(query: str) -> str:
    """根据问题动词粗略判断作答任务类型。

    这里不是训练了一个独立分类器，而是用轻量规则先分出几类最常见任务，
    这样能在本地 7B 模型条件下，以很小成本获得更稳定的作答风格控制。
    """
    cleaned = clean_text(query, limit=240)
    if not cleaned:
        return "default"

    normalized = re.sub(r"\s+", "", cleaned.lower())
    edit_markers = ("纠正", "修正", "更正", "校正", "改写", "补全", "续写", "修改", "润色", "改成", "改为")
    if any(marker in normalized for marker in edit_markers) or re.search(r"只输出[^。；\n]{0,16}(?:文本|正文|内容)", cleaned):
        return "edit"

    summary_markers = ("概括", "总结", "归纳", "概述", "核心内容", "主要内容", "简述")
    if any(marker in cleaned for marker in summary_markers):
        return "summary"

    fact_markers = ("谁", "何时", "什么时候", "哪天", "哪一年", "多少", "哪些", "哪个", "哪里", "什么")
    if any(marker in cleaned for marker in fact_markers):
        return "factoid"

    return "default"


def extract_prompt_aspect_clauses(query: str, *, max_clauses: int = 4) -> list[str]:
    """从问题中提取提示词层面的分项子句。"""
    cleaned = clean_text(query, limit=240)
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
        if len(clause) < 6:
            continue
        signature = normalize_text(clause)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        normalized.append(clause)
        if len(normalized) >= max_clauses:
            break
    return normalized


def estimate_parallel_requirement_count(query: str, *, max_count: int = 4) -> int:
    """估计问题中并列回答要求的数量。"""
    cleaned = clean_text(query, limit=240)
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


def build_task_aligned_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """按任务形态生成单轮作答提示词。

    这就是论文里“按题作答”的核心：不是多想一步，而是让模型按题目结构履约。
    """
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)

    task_mode = infer_query_task_mode(query)
    aspect_count = estimate_parallel_requirement_count(query)
    task_lines: list[str] = []
    if task_mode == "edit":
        task_lines.extend(
            [
                "- 这是文本修订任务，不是摘要任务。若原文主体和句式可被证据支持，应尽量保留，只修正错误、缺失或与证据冲突的片段。",
                "- 不要把待修订文本压缩成一句概述；应输出修正后的完整正文。",
            ]
        )
    elif task_mode == "factoid":
        task_lines.extend(
            [
                "- 这是属性定位任务。回答必须写成一整句自包含陈述，明确写出对象、属性和值，不要只输出裸日期、数字、名称或地点短语。",
                "- 若证据显示某个属性无法确定，应直接在这句中说明该属性无法从给定证据确定。",
            ]
        )
    elif task_mode == "summary":
        task_lines.append("- 这是概括任务。优先用 1 到 2 句覆盖主体、动作、结果或处理机制，避免无关背景扩写。")

    if aspect_count >= 2 and task_mode != "edit":
        aspect_clauses = extract_prompt_aspect_clauses(query, max_clauses=aspect_count)
        if aspect_clauses:
            task_lines.append("- 问题包含多个并列要求。回答必须按原顺序覆盖下列要求，不得只回答其中一部分：")
            for index, clause in enumerate(aspect_clauses, start=1):
                task_lines.append(f"  {index}. {clause}")
            task_lines.append(
                f"- 优先按上述顺序用 {len(aspect_clauses)} 个紧凑分句或句子覆盖；如果某一项证据不足，明确指出该项无法从给定证据确定。"
            )
        else:
            task_lines.append(
                f"- 问题至少包含 {aspect_count} 个并列要求。优先按原顺序用 {aspect_count} 个紧凑分句或句子覆盖，不要把它们压成笼统总结。"
            )

    task_block = "\n".join(task_lines)
    if task_block:
        task_block += "\n"

    return (
        "你是一个严格的证据问答助手。请仅根据下列证据单元回答用户问题。\n"
        "你需要先在心里判断最小充分回答形态，再输出最终答案；不要展示思考过程。\n"
        "- 回答必须直接对齐用户问题的表达框架，优先沿用问题中的主语、谓语和限定条件。\n"
        "- 回答应当自包含：即使单独读这一句，也能看出是在回答哪个对象、事件或结论。\n"
        "- 若证据中已经给出可直接填入问题的时间、数字、专名、范围或结论，优先复用原词，不做无必要改写。\n"
        "- 涉及时间、日期、数量、地点或范围时，保留证据中的完整限定形式，不要省略单位、时区、起止边界等必要信息。\n"
        f"{task_block}"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n"
        "- 不要编造证据中没有的事实，不要输出 JSON、编号、项目符号、来源说明，避免“根据资料/根据证据”这类套话。\n\n"
        f"证据单元：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]], *, style: str) -> str:
    """按指定风格统一分发提示词。"""
    if style == "task_aligned":
        return build_task_aligned_answer_prompt(query, evidence_units)
    if style == "aligned":
        return build_aligned_answer_prompt(query, evidence_units)
    return build_simple_answer_prompt(query, evidence_units)


def request_completion(
    prompt: str,
    *,
    model_name: str = DEFAULT_LLM_MODEL,
    num_predict: int = 256,
) -> str:
    """调用本地回答模型生成文本。

    这是按题作答真正落到模型调用的一层。前面的任务识别、分项估计、提示词组织，
    到这里才会变成最终的自然语言回答。
    """
    effective_prompt = apply_model_prompt_controls(model_name, prompt)
    cache_key = (model_name, num_predict, effective_prompt)
    cached = _COMPLETION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": effective_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": num_predict},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE_API,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            data = json.load(response)
        text = str(data.get("response") or "").strip()
        if text:
            _COMPLETION_CACHE[cache_key] = text
        return text
    except Exception:
        return ""


def generate_answer(prompt: str, *, model_name: str = DEFAULT_LLM_MODEL, num_predict: int = 256) -> str:
    """生成最终答案；若调用失败，则回落到统一拒答文本。"""
    answer = request_completion(prompt, model_name=model_name, num_predict=num_predict)
    return answer or NO_CONTEXT_ANSWER
