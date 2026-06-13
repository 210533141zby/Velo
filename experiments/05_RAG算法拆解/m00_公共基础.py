"""论文 RAG 主线拆解版的公共基础层。

这个文件只放所有模块都会复用的通用工具，不放论文核心创新逻辑。
这样做的目的是把“算法真正有创新的地方”和“任何检索系统都需要的基础设施”

"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "04_算法实现"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from retrieval_pipeline.common import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_RERANK_MODEL,
    EMBED_BATCH_SIZE,
    EMBEDDING_API,
    NO_CONTEXT_ANSWER,
    OLLAMA_GENERATE_API,
    QUERY_BATCH_SIZE,
    REFUSAL_MARKERS,
    SNIPPET_MAX_SENTENCES,
    SNIPPET_WINDOW_CHARS,
    SNIPPET_WINDOW_STRIDE,
    apply_model_prompt_controls,
    build_cache_signature,
    build_text_signature,
    bm25_rank,
    clean_text,
    contains_refusal,
    coverage_ratio,
    embed_texts,
    ensure_bm25_index,
    ensure_contextualized_texts,
    ensure_dir,
    ensure_doc_embeddings,
    ensure_query_embeddings,
    ensure_query_rewrites,
    hash_text,
    load_reranker,
    match_any_answer,
    mmr_select_units,
    normalize_text,
    parse_json_object,
    percentile,
    semantic_similarity_scores,
    slugify,
    text_similarity,
    tokenize_text,
)


def 公共层说明() -> str:
    """

    这个函数本身没有算法作用，只是为了让阅读者一眼明白：
    当前文件不是论文创新点，而是把其余文件依赖的公共方法收口到一起。
    """
    return (
        "本文件只负责公共能力，例如文本清洗、分词、相似度、缓存、模型调用；"
        "分项重排、覆盖取证、按题作答等论文主线方法请直接看对应主题文件。"
    )
