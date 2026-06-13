from __future__ import annotations

import re


def compact_text(value: str, limit: int | None = None) -> str:
    """压缩文本中的空白并保留核心内容。"""
    normalized = re.sub(r'\s+', ' ', str(value or '').strip())
    if limit is not None:
        return normalized[:limit].strip()
    return normalized


def compact_structured_text(value: str, limit: int | None = None) -> str:
    """压缩结构化文本中的空白。"""
    lines: list[str] = []
    pending_blank = False
    for raw_line in str(value or '').splitlines():
        normalized_line = re.sub(r'[ \t\f\v]+', ' ', raw_line).strip()
        if not normalized_line:
            if lines and not pending_blank:
                lines.append('')
                pending_blank = True
            continue
        lines.append(normalized_line)
        pending_blank = False

    normalized = '\n'.join(lines).strip()
    if limit is not None:
        return normalized[:limit].rstrip()
    return normalized


def split_text_segments(content: str) -> list[str]:
    """按规则拆分文本片段。"""
    normalized = re.sub(
        r'(?<=[\u4e00-\u9fff）】])\s*[-•]\s*(?=[\u4e00-\u9fff【（])',
        '\n',
        content or '',
    )
    return [segment.strip() for segment in re.split(r'(?<=[。！？；])|\n+', normalized) if segment.strip()]


def split_paragraphs(content: str) -> list[str]:
    """拆分段落列表。"""
    paragraphs = [paragraph.strip() for paragraph in re.split(r'\n{2,}', content or '') if paragraph.strip()]
    return paragraphs or split_text_segments(content)
