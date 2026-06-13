from __future__ import annotations

import json
import re
from typing import Any


def strip_markdown_code_fence(value: str) -> str:
    """剥离模型输出最外层的 Markdown 代码块包装。

    判别模型和综合模型有时会把 JSON 放进 ```json 代码块里，这里先把外壳去掉，
    方便后续直接解析。
    """
    stripped = str(value or '').strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```(?:[\w+-]+)?\s*', '', stripped, flags=re.IGNORECASE)
        stripped = re.sub(r'\s*```$', '', stripped)
    return stripped.strip()


def extract_json_dict(value: str) -> dict[str, Any]:
    """尽量从模型输出里提取出最外层 JSON 字典。

    当模型额外输出说明文字或前后缀时，会优先尝试截取最外层花括号区域，
    提高结构化解析成功率。
    """
    stripped = strip_markdown_code_fence(value)
    candidates = [stripped]
    start = stripped.find('{')
    end = stripped.rfind('}')
    if start != -1 and end != -1 and end > start:
        embedded = stripped[start : end + 1]
        if embedded != stripped:
            candidates.append(embedded)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}
