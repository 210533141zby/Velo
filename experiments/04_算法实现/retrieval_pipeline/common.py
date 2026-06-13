from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import jieba
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

NO_CONTEXT_ANSWER = "根据当前检索到的知识库内容，没有找到足够相关的参考资料，因此我暂时无法给出可靠回答。"
REFUSAL_MARKERS = (
    "无法确定",
    "无法回答",
    "无法给出可靠回答",
    "没有找到足够相关",
    "资料不足",
    "没有相关资料",
    "无法从提供的上下文",
    "cannot determine",
    "insufficient context",
)

EMBEDDING_API = "http://127.0.0.1:11434/v1/embeddings"
OLLAMA_GENERATE_API = "http://127.0.0.1:11434/api/generate"
DEFAULT_EMBEDDING_MODEL = "bge-m3:latest"
DEFAULT_LLM_MODEL = "qwen2.5:7b-instruct"
DEFAULT_REWRITE_MODEL = DEFAULT_LLM_MODEL
DEFAULT_RERANK_MODEL = "/root/Velo/backend/data/models/rerank/BAAI--bge-reranker-v2-m3"
EMBED_BATCH_SIZE = 48
QUERY_BATCH_SIZE = 32
SNIPPET_WINDOW_CHARS = 360
SNIPPET_WINDOW_STRIDE = 180
SNIPPET_MAX_SENTENCES = 4
_COMPLETION_CACHE: dict[tuple[str, int, str], str] = {}


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回该路径，供缓存、结果和中间文件统一复用。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(text: str, limit: int | None = None) -> str:
    """压缩空白、兜底空值，并在需要时裁剪文本长度。

    实验里的文档来自不同数据源，先统一清洗格式，才能让分词、索引和日志预览
    都保持稳定。
    """
    value = str(text or "").strip()
    value = re.sub(r"\s+", " ", value)
    if not value:
        value = "空白内容"
    return value[:limit] if limit is not None else value


def normalize_text(text: str) -> str:
    """把文本压成适合去重、匹配和拒答检测的归一化形式。"""
    lowered = clean_text(text).lower()
    lowered = re.sub(r"\s+", "", lowered)
    return re.sub(r"[，,。.!！？?；;：:\"'“”‘’（）()【】《》<>]", "", lowered)


def hash_text(text: str) -> str:
    """为文本生成稳定哈希，用作语料去重和缓存键的一部分。"""
    return hashlib.sha1(clean_text(text).encode("utf-8")).hexdigest()


def slugify(value: str) -> str:
    """把任意字符串转换成可安全写入文件名和缓存路径的标识。"""
    return re.sub(r"[^0-9a-zA-Z_-]+", "_", str(value or "")).strip("_").lower() or "default"


def apply_model_prompt_controls(model_name: str, prompt: str) -> str:
    """按模型特性补充额外控制，保证离线评测输出稳定。

    实验里会同时比较多种提示词组织方式，但底层模型的默认行为并不完全一致。
    这里负责在不改动实验提示词主体的前提下，补上模型级开关，例如关闭额外思维输出，
    以免不同模型默认配置干扰算法对比结论。
    """
    normalized_model = str(model_name or "").lower()
    if normalized_model.startswith("qwen3") and "/no_think" not in prompt and "/think" not in prompt:
        return f"/no_think\n{prompt}"
    return prompt


def contains_refusal(text: str) -> bool:
    """判断回答是否本质上是在拒答或说明证据不足。"""
    normalized = normalize_text(text)
    if not normalized:
        return True
    if normalized == normalize_text(NO_CONTEXT_ANSWER):
        return True
    return any(normalize_text(marker) in normalized for marker in REFUSAL_MARKERS)


def tokenize_text(text: str) -> list[str]:
    """对文本执行中文分词与基础归一化，供 BM25 和覆盖率统计复用。"""
    cleaned = clean_text(text).lower()
    tokens = [token.strip() for token in jieba.lcut(cleaned) if token.strip()]
    normalized: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            normalized.append(token)
            continue
        parts = re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]+", token)
        normalized.extend(parts if parts else [token])
    return [item for item in normalized if item]


def coverage_ratio(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    """衡量候选文本覆盖了多少查询词，是最轻量的相关性信号之一。"""
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def build_cache_signature(item_ids: Sequence[str]) -> str:
    """根据输入 id 列表生成缓存签名，用于区分不同评测批次。"""
    digest = hashlib.sha1("\n".join(item_ids).encode("utf-8")).hexdigest()
    return f"n{len(item_ids)}_{digest[:12]}"


def build_text_signature(texts: Sequence[str]) -> str:
    """根据文本内容生成签名，用于识别语料内容是否发生变化。"""
    digest = hashlib.sha1(
        "\n".join(clean_text(text, limit=256) for text in texts).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def embed_texts(model_name: str, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """批量生成文本向量，并对异常请求执行降级重试。

    这是实验检索侧最底层的向量化入口。除了正常分批请求外，
    这里还处理长文本、脏字符或单条请求失败时的递归拆批与降级裁剪，
    目的是让整批实验尽量跑完，而不是因为少量异常文本直接中断。
    """
    empty_embedding: list[float] | None = None

    def request_batch(batch_texts: list[str]) -> list[list[float]]:
        """向 embedding 服务请求一批文本的向量结果。"""
        payload = json.dumps({"model": model_name, "input": batch_texts}).encode("utf-8")
        request = urllib.request.Request(
            EMBEDDING_API,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
        return [item["embedding"] for item in data["data"]]

    def get_empty_embedding() -> list[float]:
        """在所有降级手段都失败时，生成一条空白文本向量作为兜底。"""
        nonlocal empty_embedding
        if empty_embedding is None:
            empty_embedding = request_batch(["空白内容"])[0]
        return empty_embedding

    def embed_batch_with_retry(batch_texts: list[str]) -> list[list[float]]:
        """在请求失败时自动拆批、裁剪和重试，尽量保住整批实验可继续执行。"""
        current_batch = [clean_text(text, limit=1400) for text in batch_texts]
        try:
            return request_batch(current_batch)
        except Exception:
            if len(current_batch) == 1:
                fallback_candidates = [
                    current_batch[0][:900],
                    re.sub(r"[^\w\u4e00-\u9fff，。！？；：、“”‘’（）()【】《》\- ]+", " ", current_batch[0])[:700],
                    re.sub(r"\s+", " ", current_batch[0])[:400],
                    "空白内容",
                ]
                for candidate in fallback_candidates:
                    try:
                        return request_batch([clean_text(candidate, limit=700)])
                    except Exception:
                        continue
                return [get_empty_embedding()]
            midpoint = max(1, len(current_batch) // 2)
            return embed_batch_with_retry(current_batch[:midpoint]) + embed_batch_with_retry(current_batch[midpoint:])

    embeddings: list[list[float]] = []
    effective_batch_size = 16 if model_name.startswith("bge-m3") else batch_size
    for start in range(0, len(texts), effective_batch_size):
        embeddings.extend(embed_batch_with_retry(texts[start : start + effective_batch_size]))
    array = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def ensure_doc_embeddings(
    cache_root: Path,
    dataset_name: str,
    model_name: str,
    corpus_ids: list[str],
    corpus_texts: list[str],
) -> np.ndarray:
    """确保语料文档向量缓存存在；若缓存缺失，则重新编码并落盘。"""
    model_slug = slugify(model_name)
    cache_signature = build_cache_signature(corpus_ids)
    text_signature = build_text_signature(corpus_texts)
    cache_path = ensure_dir(cache_root) / f"{dataset_name}_{model_slug}_{cache_signature}_{text_signature}_doc_embeddings.npy"
    if cache_path.exists():
        return np.load(cache_path)
    embeddings = embed_texts(model_name, corpus_texts)
    np.save(cache_path, embeddings)
    return embeddings


def ensure_query_embeddings(
    cache_root: Path,
    dataset_name: str,
    model_name: str,
    query_ids: list[str],
    queries: list[str],
    *,
    tag: str,
) -> np.ndarray:
    """确保查询向量缓存存在；若缓存缺失，则重新编码并落盘。"""
    model_slug = slugify(model_name)
    cache_signature = build_cache_signature(query_ids)
    text_signature = build_text_signature(queries)
    cache_path = ensure_dir(cache_root) / f"{dataset_name}_{tag}_{model_slug}_{cache_signature}_{text_signature}_query_embeddings.npy"
    if cache_path.exists():
        return np.load(cache_path)
    embeddings = embed_texts(model_name, queries)
    np.save(cache_path, embeddings)
    return embeddings


def parse_json_object(text: str) -> dict[str, Any]:
    """尽量从模型返回文本中剥离并解析出最外层 JSON 对象。

    实验里有些路线要求模型返回结构化 JSON，但实际输出可能带代码块或前后缀，
    这里负责做最大兼容。
    """
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            text = object_match.group(0)
    decoder = json.JSONDecoder()
    for start_index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start_index:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return json.loads(text)


def request_query_rewrite(query: str, model_name: str = DEFAULT_REWRITE_MODEL) -> dict[str, str]:
    """请求模型生成检索侧使用的问题改写结果。"""
    prompt = (
        "请把下面的检索问题改写成适合混合检索的 JSON，只输出 JSON。\n"
        '格式严格为 {"sparse":"...","dense":"..."}。\n'
        "要求：\n"
        "1. sparse 用空格分隔，保留原问题中的核心实体、限定条件和关键属性，适合词项检索。\n"
        "2. dense 保留原问题语义与约束，改写成简洁完整的自然语言检索句，适合向量检索。\n"
        "3. 不要编造事实，不要丢失原问题中的时间、数量、关系或对象。\n"
        f"问题：{query}"
    )
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 120},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE_API,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.load(response)
        parsed = parse_json_object(str(data.get("response") or ""))
        sparse = clean_text(str(parsed.get("sparse") or query), limit=96)
        dense = clean_text(str(parsed.get("dense") or query), limit=120)
        return {"sparse": sparse or query, "dense": dense or query}
    except Exception:
        return {"sparse": clean_text(query, limit=96), "dense": clean_text(query, limit=120)}


def ensure_query_rewrites(
    cache_root: Path,
    dataset_name: str,
    query_ids: list[str],
    queries: list[str],
    *,
    model_name: str = DEFAULT_REWRITE_MODEL,
) -> list[dict[str, str]]:
    """确保查询改写结果缓存存在。

    查询改写属于检索侧对比实验的一条独立变量，不应该每次重跑都重新生成，
    否则很难保证结果可复现。这里把每条问题的 sparse / dense 改写缓存到磁盘，
    让后续实验直接复用同一批改写结果。
    """
    cache_signature = build_cache_signature(query_ids)
    cache_path = ensure_dir(cache_root) / f"{dataset_name}_{cache_signature}_query_rewrites.json"
    cached: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        raw_items = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = {
            str(item["query_id"]): {
                "sparse": str(item.get("sparse") or ""),
                "dense": str(item.get("dense") or ""),
            }
            for item in raw_items
        }

    changed = False
    results: list[dict[str, str]] = []
    for query_id, query in zip(query_ids, queries):
        rewrite = cached.get(query_id)
        if rewrite is None or not rewrite.get("sparse") or not rewrite.get("dense"):
            rewrite = request_query_rewrite(query, model_name=model_name)
            cached[query_id] = rewrite
            changed = True
        results.append({"query_id": query_id, **rewrite})

    if changed:
        serialized = [{"query_id": query_id, **cached[query_id]} for query_id in query_ids]
        cache_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def ensure_bm25_index(
    cache_root: Path,
    dataset_name: str,
    corpus_ids: list[str],
    corpus_texts: list[str],
) -> tuple[BM25Okapi, list[list[str]]]:
    """确保 BM25 索引缓存存在。

    稀疏检索是实验主干的一半。为了避免每次运行都重新分词并构建 BM25，
    这里把语料分词结果和索引对象一起缓存下来，保证对比实验能快速复跑，
    也便于确认“检索差异来自策略变化，而不是临时重建索引带来的随机扰动”。
    """
    cache_signature = build_cache_signature(corpus_ids)
    text_signature = build_text_signature(corpus_texts)
    tokens_path = ensure_dir(cache_root) / f"{dataset_name}_{cache_signature}_{text_signature}_corpus_tokens.pkl"
    bm25_path = ensure_dir(cache_root) / f"{dataset_name}_{cache_signature}_{text_signature}_bm25.pkl"
    if bm25_path.exists() and tokens_path.exists():
        with tokens_path.open("rb") as handle:
            corpus_tokens = pickle.load(handle)
        with bm25_path.open("rb") as handle:
            bm25 = pickle.load(handle)
        return bm25, corpus_tokens

    if tokens_path.exists():
        with tokens_path.open("rb") as handle:
            corpus_tokens = pickle.load(handle)
    else:
        corpus_tokens = [tokenize_text(text) for text in corpus_texts]
        with tokens_path.open("wb") as handle:
            pickle.dump(corpus_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

    bm25 = BM25Okapi(corpus_tokens)
    with bm25_path.open("wb") as handle:
        pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return bm25, corpus_tokens


def load_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoder:
    """加载实验使用的重排模型。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_path = Path(model_name)
    resolved_source = str(local_path) if local_path.exists() else model_name
    return CrossEncoder(resolved_source, device=device, max_length=512, local_files_only=local_path.exists())


def build_light_context_prefix(*, title: str, text: str) -> str:
    """为上下文化检索生成轻量级文档前缀。"""
    sentences = [segment.strip() for segment in re.split(r"(?<=[。！？!?；;])\s*", clean_text(text, limit=900)) if segment.strip()]
    lead_sentences = " ".join(sentences[:2]) if sentences else clean_text(text, limit=180)
    title_text = clean_text(title, limit=120)
    if title_text and title_text not in lead_sentences:
        return clean_text(f"{title_text} {lead_sentences}", limit=220)
    return clean_text(lead_sentences, limit=220)


def ensure_contextualized_texts(
    cache_root: Path,
    dataset_name: str,
    corpus_ids: list[str],
    titles: list[str],
    corpus_texts: list[str],
    *,
    model_name: str = DEFAULT_LLM_MODEL,
) -> list[str]:
    """确保带上下文前缀的文档文本缓存存在。

    这对应论文里“上下文化检索”的检索侧预实验。它不会改动原始文档内容，
    而是在前面补一个轻量的标题/前情摘要前缀，用来测试“给文档块增加背景信息”
    是否能稳定提升召回与作答质量。
    """
    cache_signature = build_cache_signature(corpus_ids)
    text_signature = build_text_signature(corpus_texts)
    cache_path = ensure_dir(cache_root) / f"{dataset_name}_{cache_signature}_{text_signature}_light_contextualized.json"
    cached: dict[str, str] = {}
    if cache_path.exists():
        cached = {
            str(item["corpus_id"]): str(item.get("contextual_text") or "")
            for item in json.loads(cache_path.read_text(encoding="utf-8"))
        }

    changed = False
    results: list[str] = []
    for corpus_id, title, text in zip(corpus_ids, titles, corpus_texts):
        contextual_text = cached.get(corpus_id, "")
        if not contextual_text:
            raw_text = clean_text(text, limit=1600)
            prefix = build_light_context_prefix(title=title, text=text)
            contextual_text = clean_text(f"{prefix}\n{raw_text}", limit=1800)
            cached[corpus_id] = contextual_text
            changed = True
        results.append(contextual_text)

    if changed:
        serialized = [
            {"corpus_id": corpus_id, "contextual_text": cached.get(corpus_id, "")}
            for corpus_id in corpus_ids
        ]
        cache_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def reciprocal_rank_fusion(rankings: list[list[int]], *, top_k: int = 50, k: int = 60) -> dict[int, float]:
    """对多路检索排序执行 RRF 融合，得到统一候选优先级。

    这是实验检索主干里连接稠密检索和 BM25 的核心步骤，用来避免某一路分数
    尺度过大时直接压制另一条召回链。
    """
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_index in enumerate(ranking[:top_k], start=1):
            fused[doc_index] += 1.0 / (k + rank)
    return dict(fused)


def ranked_doc_indices(scores: dict[int, float]) -> list[int]:
    """把分数字典整理成降序文档下标列表。"""
    return [doc_index for doc_index, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def compute_dense_rankings(
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """批量计算查询对语料库的稠密检索排序结果。

    这里会自动根据当前环境切换到 GPU 或 CPU，并按批处理查询向量，
    供后续与 BM25 结果做融合。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    doc_tensor = torch.tensor(doc_embeddings, dtype=torch.float32, device=device)
    effective_top_k = max(1, min(top_k, int(doc_embeddings.shape[0])))
    top_indices_batches: list[np.ndarray] = []
    top_scores_batches: list[np.ndarray] = []
    for start in range(0, len(query_embeddings), QUERY_BATCH_SIZE):
        query_tensor = torch.tensor(query_embeddings[start : start + QUERY_BATCH_SIZE], dtype=torch.float32, device=device)
        scores = query_tensor @ doc_tensor.T
        top_scores, top_indices = torch.topk(scores, k=effective_top_k, dim=1)
        top_indices_batches.append(top_indices.cpu().numpy())
        top_scores_batches.append(top_scores.cpu().numpy())
    return np.vstack(top_indices_batches), np.vstack(top_scores_batches)


def bm25_rank(query_tokens: list[str], bm25: BM25Okapi, *, top_k: int) -> tuple[list[int], list[float]]:
    """执行 BM25 检索，并返回命中文档下标及对应原始分数。"""
    scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
    if len(scores) <= top_k:
        order = np.argsort(-scores)
    else:
        candidate = np.argpartition(-scores, top_k - 1)[:top_k]
        order = candidate[np.argsort(-scores[candidate])]
    return order.tolist(), scores[order].tolist()


def build_snippet_windows(text: str) -> list[str]:
    """把长文档切成可供重排的候选片段窗口。"""
    cleaned = clean_text(text, limit=1600)
    if len(cleaned) <= SNIPPET_WINDOW_CHARS:
        return [cleaned]

    segments = [segment.strip() for segment in re.split(r"(?<=[。！？!?；;])|\n+", cleaned) if segment.strip()]
    windows: list[str] = []
    if segments:
        for start in range(len(segments)):
            current: list[str] = []
            current_length = 0
            for end in range(start, min(len(segments), start + SNIPPET_MAX_SENTENCES)):
                segment = segments[end]
                next_length = current_length + len(segment) + (1 if current else 0)
                if current and next_length > SNIPPET_WINDOW_CHARS:
                    break
                current.append(segment)
                current_length = next_length
                if current_length >= 80:
                    windows.append(" ".join(current))

    for start in range(0, len(cleaned), SNIPPET_WINDOW_STRIDE):
        window = cleaned[start : start + SNIPPET_WINDOW_CHARS]
        if window:
            windows.append(window)
        if start + SNIPPET_WINDOW_CHARS >= len(cleaned):
            break

    deduped: list[str] = []
    seen: set[str] = set()
    for window in windows:
        normalized = window.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped or [cleaned[:SNIPPET_WINDOW_CHARS]]


def score_snippet_window(
    query_tokens: list[str],
    focus_tokens: list[str],
    window: str,
    *,
    index: int,
    total_windows: int,
) -> float:
    """计算单个片段窗口的相关性分数。"""
    query_token_set = set(query_tokens)
    focus_token_set = set(focus_tokens)
    window_token_set = set(tokenize_text(window))
    coverage = coverage_ratio(query_token_set, window_token_set)
    focus_coverage = coverage_ratio(focus_token_set, window_token_set)
    exact_hits = 0
    lowered_window = window.lower()
    for token in focus_token_set:
        if len(token) >= 2 and token in lowered_window:
            exact_hits += 1
    position_bonus = max(0.0, 1.0 - (index / max(total_windows, 1))) * 0.08
    return coverage * 0.95 + focus_coverage * 1.15 + min(exact_hits, 4) * 0.08 + position_bonus


def rank_query_focused_snippets(query_tokens: list[str], focus_tokens: list[str], text: str) -> list[str]:
    """按查询相关性切分并排序文档片段。"""
    windows = build_snippet_windows(text)
    if len(windows) == 1:
        return windows
    scored_windows = [
        (
            score_snippet_window(query_tokens, focus_tokens, window, index=index, total_windows=len(windows)),
            index,
            window,
        )
        for index, window in enumerate(windows)
    ]
    scored_windows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [window for _score, _index, window in scored_windows]


def rerank_documents(
    reranker: CrossEncoder,
    *,
    query: str,
    query_tokens: list[str],
    focus_tokens: list[str],
    candidate_doc_indices: list[int],
    corpus_texts: list[str],
    multi_snippet_count: int,
    aggregation_mode: str = "default",
    aspect_token_sets: Sequence[set[str]] | None = None,
) -> tuple[list[int], dict[int, list[str]], dict[int, float]]:
    """调用重排模型对候选文档重新排序。

    这个函数承担实验链路里“融合召回之后再做精排”的职责。它不仅返回最终文档顺序，
    还会把每篇文档对应的代表性片段和聚合后的文档分数一并返回，
    这样后面的证据构造阶段就不需要再次回头切片。

    当实验变体启用 aspect-aware 模式时，这里还会把“覆盖多个问题分项”的能力
    融到聚合打分里，用来模拟论文中分项重排前的精排偏好。
    """
    if not candidate_doc_indices:
        return [], {}, {}

    query_token_set = set(query_tokens)
    effective_aspect_token_sets = [set(tokens) for tokens in (aspect_token_sets or []) if tokens]
    pair_doc_indices: list[int] = []
    pairs: list[tuple[str, str]] = []
    doc_snippets: dict[int, list[str]] = {}
    snippet_records: list[dict[str, Any]] = []
    for doc_index in candidate_doc_indices:
        ranked_snippets = rank_query_focused_snippets(query_tokens, focus_tokens, corpus_texts[doc_index])
        selected_snippets = ranked_snippets[: max(1, multi_snippet_count)]
        doc_snippets[int(doc_index)] = selected_snippets
        for snippet in selected_snippets:
            snippet_tokens = set(tokenize_text(snippet))
            base_coverage = coverage_ratio(query_token_set, snippet_tokens)
            detail_signal = 1.0 if re.search(r"\d|[%年月日:：\-]|[A-Z]{2,}", snippet) else 0.0
            covered_aspect_ids = {
                aspect_index
                for aspect_index, aspect_tokens in enumerate(effective_aspect_token_sets)
                if coverage_ratio(aspect_tokens, snippet_tokens) >= 0.14
            }
            pair_doc_indices.append(int(doc_index))
            pairs.append((query, snippet))
            snippet_records.append(
                {
                    "doc_index": int(doc_index),
                    "snippet": snippet,
                    "tokens": snippet_tokens,
                    "base_coverage": base_coverage,
                    "detail_signal": detail_signal,
                    "covered_aspect_ids": covered_aspect_ids,
                }
            )

    raw_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    if aggregation_mode in {"aspect_aware", "aspect_aware_conservative"} and len(effective_aspect_token_sets) >= 2:
        grouped_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
        is_conservative = aggregation_mode == "aspect_aware_conservative"
        clause_threshold = 0.22 if is_conservative else 0.14
        for record, raw_score in zip(snippet_records, raw_scores.tolist()):
            ce_score = 1.0 / (1.0 + math.exp(-float(raw_score)))
            covered_aspect_ids = set(record["covered_aspect_ids"])
            aspect_ratio = len(covered_aspect_ids) / len(effective_aspect_token_sets)
            if is_conservative:
                covered_aspect_ids = {
                    aspect_index
                    for aspect_index, aspect_tokens in enumerate(effective_aspect_token_sets)
                    if coverage_ratio(aspect_tokens, record["tokens"]) >= clause_threshold
                }
                aspect_ratio = len(covered_aspect_ids) / len(effective_aspect_token_sets)
                blended_score = (
                    0.80 * ce_score
                    + 0.10 * float(record["base_coverage"])
                    + 0.06 * aspect_ratio
                    + 0.04 * float(record["detail_signal"])
                )
            else:
                blended_score = (
                    0.68 * ce_score
                    + 0.14 * float(record["base_coverage"])
                    + 0.12 * aspect_ratio
                    + 0.06 * float(record["detail_signal"])
                )
            grouped_records[int(record["doc_index"])].append(
                {
                    **record,
                    "covered_aspect_ids": covered_aspect_ids,
                    "ce_score": ce_score,
                    "aspect_ratio": aspect_ratio,
                    "blended_score": blended_score,
                }
            )

        aggregated_scores: dict[int, float] = {}
        for doc_index, records in grouped_records.items():
            snippet_sort_key = "ce_score" if is_conservative else "blended_score"
            records.sort(key=lambda item: float(item[snippet_sort_key]), reverse=True)
            doc_snippets[int(doc_index)] = [str(item["snippet"]) for item in records]
            best_score = max(float(item["blended_score"]) for item in records)
            avg_score = sum(float(item["blended_score"]) for item in records) / len(records)
            best_coverage = max(float(item["base_coverage"]) for item in records)
            best_detail = max(float(item["detail_signal"]) for item in records)
            covered_aspects = set().union(*(item["covered_aspect_ids"] for item in records))
            doc_aspect_ratio = len(covered_aspects) / len(effective_aspect_token_sets)
            if is_conservative:
                best_ce = max(float(item["ce_score"]) for item in records)
                avg_ce = sum(float(item["ce_score"]) for item in records) / len(records)
                aggregated_scores[int(doc_index)] = (
                    0.74 * best_ce
                    + 0.18 * avg_ce
                    + 0.06 * doc_aspect_ratio
                    + 0.02 * best_coverage
                )
            else:
                aggregated_scores[int(doc_index)] = (
                    0.56 * best_score
                    + 0.18 * avg_score
                    + 0.18 * doc_aspect_ratio
                    + 0.05 * best_coverage
                    + 0.03 * best_detail
                )
        ranked = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_index for doc_index, _ in ranked], doc_snippets, aggregated_scores

    max_score_map: dict[int, float] = {}
    sum_score_map: dict[int, float] = {}
    count_score_map: dict[int, int] = {}
    for doc_index, score in zip(pair_doc_indices, raw_scores.tolist()):
        numeric_score = float(score)
        max_score_map[doc_index] = max(max_score_map.get(doc_index, float("-inf")), numeric_score)
        sum_score_map[doc_index] = sum_score_map.get(doc_index, 0.0) + numeric_score
        count_score_map[doc_index] = count_score_map.get(doc_index, 0) + 1

    aggregated_scores = {
        doc_index: 0.85 * max_score_map[doc_index] + 0.15 * (sum_score_map[doc_index] / count_score_map[doc_index])
        for doc_index in max_score_map
    }
    ranked = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_index for doc_index, _ in ranked], doc_snippets, aggregated_scores


def jaccard_similarity(left_tokens: set[str], right_tokens: set[str]) -> float:
    """计算两个词集合的 Jaccard 相似度。"""
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def mmr_select_units(
    units: list[dict[str, Any]],
    *,
    limit: int,
    lambda_weight: float = 0.72,
) -> list[dict[str, Any]]:
    """用 MMR 策略挑选互补证据单元。

    该函数主要用于对照实验，测试“在高分片段之间显式去冗余”能否替代论文方案中的
    更细粒度证据排序。实现上先尽量保留每篇文档最强的一条证据，再用贪心 MMR
    在相关性和互补性之间做平衡。
    """
    def greedy_select(candidates: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """用贪心方式继续挑选互补证据。"""
        remaining = list(candidates)
        chosen = list(selected)
        while remaining and len(chosen) < limit:
            best_unit: dict[str, Any] | None = None
            best_value = float("-inf")
            for unit in remaining:
                relevance = float(unit["score"])
                redundancy = 0.0
                if chosen:
                    redundancy = max(
                        jaccard_similarity(set(unit["tokens"]), set(existing["tokens"])) for existing in chosen
                    )
                mmr_value = lambda_weight * relevance - (1.0 - lambda_weight) * redundancy
                if mmr_value > best_value:
                    best_value = mmr_value
                    best_unit = unit
            if best_unit is None:
                break
            chosen.append(best_unit)
            remaining = [item for item in remaining if item is not best_unit]
        return chosen

    best_by_doc: dict[int, dict[str, Any]] = {}
    secondary_units: list[dict[str, Any]] = []
    for unit in units:
        doc_id = int(unit["doc_id"])
        current_best = best_by_doc.get(doc_id)
        if current_best is None:
            best_by_doc[doc_id] = unit
            continue
        if float(unit["score"]) > float(current_best["score"]):
            secondary_units.append(current_best)
            best_by_doc[doc_id] = unit
        else:
            secondary_units.append(unit)

    primary_units = sorted(best_by_doc.values(), key=lambda item: float(item["score"]), reverse=True)
    selected = greedy_select(primary_units, [])
    if len(selected) >= limit:
        return selected[:limit]

    remaining_units = sorted(secondary_units, key=lambda item: float(item["score"]), reverse=True)
    return greedy_select(remaining_units, selected)[:limit]


def build_simple_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """生成最简直接作答基线的提示词。

    这条提示词几乎不引入额外结构约束，主要用于回答侧基线，
    帮助对比后续各种对齐式、忠实式和结构化提示能带来多大提升。
    """
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    return (
        "你是一个问答助手。请仅根据下列证据单元回答用户问题。\n"
        f"- 第一句必须直接回答问题。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n"
        "- 不要编造证据中没有的事实，不要输出 JSON、编号或来源说明。\n"
        "- 如果问题需要多个事实点，按清晰顺序给出，但保持简洁。\n\n"
        f"证据单元：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_neutral_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """生成语气更中性、约束适中的回答提示词。

    该版本用于观察在不强推特定结构的前提下，仅靠适度忠实性约束时的作答效果。
    """
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


def build_model_native_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """生成更接近模型原生问答习惯的提示词。

    这条路线会尽量少加人为模板，用来测试“放松提示词控制后模型自然作答”
    与主线方案之间的差异。
    """
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    return (
        "请阅读证据并直接回答问题。\n"
        "要求：\n"
        "1. 只依据证据回答。\n"
        "2. 优先保留证据中的关键时间、数字、专名和限定条件。\n"
        "3. 若问题涉及多个方面，用自然语言完整覆盖，不要补充无关背景。\n"
        "4. 若证据无法确定答案，明确说明无法从给定证据确定。\n\n"
        f"证据：\n{context}\n\n"
        f"问题：{query}\n"
        "请直接作答："
    )


def build_faithful_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """生成强调证据忠实性的回答提示词。

    它要求模型尽量沿用证据中的关键原词、数值和限定条件，
    用来观察“更强忠实约束”对复杂问答质量的影响。
    """
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    return (
        "你是一个严格依据证据作答的问答助手。请仅根据下列证据单元回答用户问题。\n"
        "- 第一句必须直接回答问题，并优先复用证据中的关键原词，尤其是专名、时间、数字、范围和结论。\n"
        "- 只保留完成回答所必需的信息，不要扩写背景，不要加入证据中没有的解释。\n"
        "- 如果问题要求多个事实点，按问题本身的顺序逐一回答；可以分句，但保持紧凑。\n"
        "- 如果问题要求概括，只输出 1 到 2 句核心内容。\n"
        "- 如果问题要求纠正、改写或补全文本，只输出处理后的正文。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n"
        "- 不要输出 JSON、编号、项目符号、来源说明，也不要使用“根据资料”等套话。\n\n"
        f"证据单元：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_aligned_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """生成与问题表达框架强对齐的作答提示词。

    这是实验主线里最接近论文最终回答阶段的一类提示，
    重点不是多想一步，而是让模型在回答时始终贴着问题的槽位和限定条件组织内容。
    """
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
        "- 如果问题要求概括，只输出 1 到 2 句核心摘要。\n"
        "- 如果问题要求纠正、改写或补全文本，只输出修正后的正文，不解释。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n"
        "- 不要编造证据中没有的事实，不要输出 JSON、编号、项目符号、来源说明，避免“根据资料/根据证据”这类套话。\n\n"
        f"证据单元：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def infer_query_task_mode(query: str) -> str:
    """根据问题动词粗略判断作答任务类型。"""
    cleaned = clean_text(query, limit=240)
    if not cleaned:
        return "default"

    normalized = re.sub(r"\s+", "", cleaned.lower())
    edit_markers = (
        "纠正",
        "修正",
        "更正",
        "校正",
        "改写",
        "补全",
        "续写",
        "修改",
        "润色",
        "改成",
        "改为",
    )
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
    """按任务形态生成单轮作答提示词。"""
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
        task_lines.extend(
            [
                "- 这是概括任务。优先用 1 到 2 句覆盖主体、动作、结果或处理机制，避免无关背景扩写。",
            ]
        )

    if aspect_count >= 2 and task_mode != "edit":
        aspect_clauses = extract_prompt_aspect_clauses(query, max_clauses=aspect_count)
        if aspect_clauses:
            task_lines.append("- 问题包含多个并列要求。回答必须按原顺序覆盖下列要求，不得只回答其中一部分：")
            for index, clause in enumerate(aspect_clauses, start=1):
                task_lines.append(f"  {index}. {clause}")
            task_lines.append(
                f"- 优先按上述顺序用 {len(aspect_clauses)} 个紧凑分句或句子覆盖；如果某一项证据不足，明确指出该项无法从给定证据确定。"
            )
            task_lines.append("- 如果证据单元中已经标出“对应要求”，优先按照这些对应关系组织答案。")
            task_lines.append("- 若问题要求列举内容、措施、原因、产品、企业、条件或数量，不要只做笼统概括；应保留关键条目和总量信息。")
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


def build_needled_answer_prompt(query: str, evidence_units: Sequence[dict[str, Any]]) -> str:
    """生成带关键句标注的作答提示词。

    这条提示词服务于“关键句针扎”对照路线，测试显式高亮关键证据句
    是否足以替代更完整的证据重排与覆盖设计。
    """
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    return (
        "你是一个严格的证据问答助手。请仅根据下列证据单元回答用户问题。\n"
        "部分证据片段已用 [need_X]...[/need_X] 做了高亮，这些高亮表示它们与问题中的关键需求点直接相关。\n"
        "- 先优先利用高亮片段覆盖问题需要的关键信息点，再用原文中的必要上下文补足限定条件。\n"
        "- 回答必须直接对齐用户问题的表达框架，优先沿用问题中的主语、谓语和限定条件。\n"
        "- 若高亮片段中已经给出时间、数字、专名、范围或结论，优先复用原词，不做无必要改写。\n"
        "- 不要逐条解释高亮标签，也不要输出 need 编号；只输出自然、连贯的最终答案。\n"
        "- 如果问题需要多个事实点，按问题涉及的维度依次作答，每句只保留一个清晰信息点。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n"
        "- 不要编造证据中没有的事实，不要输出 JSON、编号、项目符号、来源说明，避免“根据资料/根据证据”这类套话。\n\n"
        f"证据单元：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_aligned_answer_prompt_with_constraints(
    query: str,
    evidence_units: Sequence[dict[str, Any]],
    constrained_spans: Sequence[str],
) -> str:
    """生成带关键片段约束的对齐作答提示词。

    该版本会把必须尽量保留的片段显式告诉模型，用来测试“限制模型改写幅度”
    是否能稳定改善答案忠实度。
    """
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"[{index}] {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    span_block = "；".join(span for span in constrained_spans if span.strip())
    return (
        "你是一个严格的证据问答助手。请仅根据下列证据单元回答用户问题。\n"
        "你需要先在心里判断最小充分回答形态，再输出最终答案；不要展示思考过程。\n"
        "- 回答必须直接对齐用户问题的表达框架，优先沿用问题中的主语、谓语和限定条件。\n"
        "- 回答应当自包含：即使单独读这一句，也能看出是在回答哪个对象、事件或结论。\n"
        "- 若证据中已经给出可直接填入问题的时间、数字、专名或结论，优先复用原词，不做无必要改写。\n"
        "- 以下关键片段若与问题相关，回答时应尽量原样保留，不要丢失或泛化改写。\n"
        f"- 关键片段：{span_block or '无'}\n"
        "- 如果一个短事实就能回答，优先写成一整句自包含陈述，不补充无关背景。\n"
        "- 如果问题需要多个事实点，按问题涉及的维度依次作答，每句只保留一个清晰信息点。\n"
        f"- 如果证据不足以得到可靠答案，原样回答：{NO_CONTEXT_ANSWER}\n"
        "- 不要编造证据中没有的事实，不要输出 JSON、编号、项目符号、来源说明，避免“根据资料/根据证据”这类套话。\n\n"
        f"证据单元：\n{context}\n\n"
        f"用户问题：{query}\n"
        "回答："
    )


def build_trace_structured_answer_prompt(
    query: str,
    evidence_units: Sequence[dict[str, Any]],
) -> str:
    """生成 TRACE 风格的结构化作答提示词。

    它要求模型显式输出相关证据、分析过程和最终答案三个部分，
    用于对比更重的结构化回答模板与本文主线方案之间的效果与成本差异。
    """
    blocks = []
    for index, unit in enumerate(evidence_units, start=1):
        blocks.append(f"{index}. {unit['title']}\n{unit['text']}")
    context = "\n\n".join(blocks)
    return (
        "你是一个透明证据问答助手。请根据给定参考资料回答问题，并显式展示“用了哪些参考资料、如何组织推理、最终答案是什么”。\n"
        "你的回答必须严格包含以下三个部分，并按顺序输出：\n\n"
        "<relevance>\n"
        "[只输出有帮助的参考资料编号列表，例如 [1,3,4]；如果都无关则输出 []]\n"
        "</relevance>\n\n"
        "<analysis>\n"
        "[结合相关参考资料进行简洁分析；说明关键结论分别来自哪些编号；不要编造参考资料中没有的事实]\n"
        "</analysis>\n\n"
        "<answer>\n"
        "[输出最终答案。答案必须是完整、自然、可单独阅读的中文回答；如果问题涉及多个维度，按问题要求覆盖这些维度；不要输出项目符号；如果证据不足，原样回答："
        f"{NO_CONTEXT_ANSWER}"
        "]\n"
        "</answer>\n\n"
        f"<question>\n{query}\n</question>\n\n"
        f"<references>\n{context}\n</references>\n"
    )


def build_answer_prompt(
    query: str,
    evidence_units: Sequence[dict[str, Any]],
    *,
    style: str,
) -> str:
    """根据指定风格构建答案提示词。

    这里是实验生成侧的统一分发入口。不同实验路线虽然证据处理方式不同，
    但最终都会在这里切换到对应的提示词风格，以保证“实验变量只改该改的那一层”，
    而不是每个脚本自己拼一套 prompt。
    """
    if style == "model_native":
        return build_model_native_answer_prompt(query, evidence_units)
    if style == "neutral":
        return build_neutral_answer_prompt(query, evidence_units)
    if style == "needled":
        return build_needled_answer_prompt(query, evidence_units)
    if style == "task_aligned":
        return build_task_aligned_answer_prompt(query, evidence_units)
    if style == "trace_structured":
        return build_trace_structured_answer_prompt(query, evidence_units)
    if style == "aligned":
        return build_aligned_answer_prompt(query, evidence_units)
    if style == "faithful":
        return build_faithful_answer_prompt(query, evidence_units)
    return build_simple_answer_prompt(query, evidence_units)


def request_completion(
    prompt: str,
    *,
    model_name: str = DEFAULT_LLM_MODEL,
    num_predict: int = 256,
) -> str:
    """调用生成接口获取文本补全结果。"""
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


def generate_json_payload(
    prompt: str,
    *,
    model_name: str = DEFAULT_LLM_MODEL,
    num_predict: int = 512,
) -> dict[str, Any]:
    """生成JSON 结果请求载荷。"""
    raw_text = request_completion(prompt, model_name=model_name, num_predict=num_predict)
    if not raw_text:
        return {}
    try:
        return parse_json_object(raw_text)
    except Exception:
        return {}


def extract_tag_text(text: str, tag_name: str) -> str:
    """提取指定标签包裹的文本内容。"""
    pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    match = re.search(pattern, str(text or ""), re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def generate_answer(prompt: str, *, model_name: str = DEFAULT_LLM_MODEL, num_predict: int = 256) -> str:
    """生成答案。"""
    answer = request_completion(prompt, model_name=model_name, num_predict=num_predict)
    return answer or NO_CONTEXT_ANSWER
