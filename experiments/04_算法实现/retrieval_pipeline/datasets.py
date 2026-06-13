"""加载论文实验使用的 RGB 与 CRUD 数据。

这个模块是实验数据层的统一入口，`experiments/02_消融实验`
和 `experiments/03_证伪实验` 下的脚本都会通过这里读取样例、
整理语料并生成可直接评测的标准结构。
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import clean_text, ensure_dir, hash_text

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "experiments" / "01_数据集与样本"
RGB_DATA_PATH = DATA_ROOT / "RGB数据" / "zh_refine.json"
CRUD_ROOT = DATA_ROOT / "CRUD数据源"
CRUD_OFFICIAL_ROOT = CRUD_ROOT / "crud_rag_official"
CRUD_SPLIT_PATH = CRUD_OFFICIAL_ROOT / "data" / "crud_split" / "split_merged.json"
CRUD_DOCS_ROOT = CRUD_OFFICIAL_ROOT / "data" / "80000_docs"

RGB_RAW_URLS = (
    "https://raw.githubusercontent.com/chen700564/RGB/master/data/zh_refine.json",
    "https://raw.githubusercontent.com/chen700564/RGB/main/data/zh_refine.json",
    "https://raw.githubusercontent.com/chen700564/RGB/master/data/zh.json",
    "https://raw.githubusercontent.com/chen700564/RGB/main/data/zh.json",
)

# 部分 CRUD 多文档样例在本地语料里存在语义等价的支撑文章，
# 这些文章足以回答问题，但未写入原始 expected_doc_ids 标注。
# 主评测仍保持官方标注口径不变；这里补充的签名只用于额外的检索分析，
# 方便判断“没有命中原标注”究竟是检索失败，还是命中了等价证据。
CRUD_EQUIVALENT_EVIDENCE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "questanswer_2docs_001": (
        "《防控儿童青少年近视核心知识十条》包括：",
        "医疗机构要以儿童家长和养育人为重点",
    ),
    "questanswer_2docs_003": (
        "第5号台风“杜苏芮”",
        "国家防总办公室27日向北京、天津、河北",
    ),
}


@dataclass
class CorpusDoc:
    """实验侧统一使用的语料文档结构。

    无论原始数据来自 RGB 还是 CRUD，进入检索流水线前都会先整理成这份最小结构，
    只保留文档编号、标题、正文和数据集来源，方便后续统一建索引、检索和回查来源。
    """
    doc_id: int
    title: str
    content: str
    dataset: str


@dataclass
class ExperimentCase:
    """实验样例的标准表示。

    这个结构既服务评测，也服务检索命中率统计。除了问题、参考答案之外，
    还会额外保留正样本文本、是否应拒答、锚点文本和期望文档编号等信息，
    供消融实验、证伪实验和误差分析共用。
    """
    dataset: str
    case_id: str
    split: str
    query: str
    reference: str
    answers: tuple[str, ...]
    positive_texts: tuple[str, ...]
    should_refuse: bool = False
    anchor: str = ""
    expected_doc_ids: tuple[int, ...] = ()


def _normalize_rgb_contexts(raw_items: Sequence[Any], prefix: str) -> list[dict[str, str]]:
    """把 RGB 原始上下文整理成统一的标题-正文结构。

    RGB 的上下文字段在不同版本里可能是纯字符串，也可能是带 `title` / `text`
    的字典。这里负责兜底兼容这些差异，统一产出后续加载逻辑可以直接消费的
    `{"title": ..., "text": ...}` 结构，避免主流程里到处写字段兼容判断。
    """
    contexts: list[dict[str, str]] = []
    for idx, item in enumerate(raw_items or []):
        if isinstance(item, str):
            text = item.strip()
            if text:
                contexts.append({"title": f"{prefix}_{idx}", "text": text})
            continue
        if isinstance(item, dict):
            title = str(
                item.get("title")
                or item.get("source")
                or item.get("name")
                or f"{prefix}_{idx}"
            ).strip()
            text = str(
                item.get("text")
                or item.get("content")
                or item.get("contents")
                or item.get("body")
                or ""
            ).strip()
            if text:
                contexts.append({"title": title or f"{prefix}_{idx}", "text": text})
    return contexts


def _flatten_rgb_answers(raw_answer: Any) -> list[str]:
    """把 RGB 的嵌套答案字段展开成一维字符串列表。

    RGB 数据里的答案有时是单字符串，有时是列表，甚至可能是多层嵌套列表。
    为了让后面的样例构造和参考答案拼接逻辑保持稳定，这里先把它递归展开，
    并顺手去掉空答案。
    """
    if isinstance(raw_answer, str):
        return [raw_answer.strip()] if raw_answer.strip() else []
    if isinstance(raw_answer, Sequence):
        answers: list[str] = []
        for item in raw_answer:
            answers.extend(_flatten_rgb_answers(item))
        return answers
    text = str(raw_answer or "").strip()
    return [text] if text else []


def _download_rgb_dataset(path: Path) -> Path:
    """确保本地存在可用的 RGB 数据文件。

    论文实验默认优先使用本地缓存；如果目标文件不存在或为空，
    才会尝试从预设的 RGB 仓库地址拉取。这样既能减少重复下载，
    也能保证实验脚本在新环境中首次运行时具备自恢复能力。
    """
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size > 0:
        return path
    for url in RGB_RAW_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                text = response.read().decode("utf-8")
            path.write_text(text, encoding="utf-8")
            return path
        except Exception:
            continue
    raise RuntimeError("RGB 数据集下载失败。")


def load_rgb_cases(*, case_limit: int = 0) -> tuple[list[ExperimentCase], list[CorpusDoc]]:
    """加载 RGB 样例，并同步构建对应语料集。

    这个入口会完成三件事：
    1. 读取并解析 RGB 原始文件；
    2. 把问题、答案、正负上下文整理成 `ExperimentCase`；
    3. 从上下文里去重构造 `CorpusDoc`，供后续检索实验直接使用。

    返回值中的 case 列表用于问答评测，doc 列表用于建立 RGB 检索语料。
    """
    path = _download_rgb_dataset(RGB_DATA_PATH)
    raw_text = path.read_text(encoding="utf-8").strip()
    try:
        raw_data = json.loads(raw_text)
        if not isinstance(raw_data, list):
            raw_data = [raw_data]
    except json.JSONDecodeError:
        raw_data = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    raw_cases: list[dict[str, Any]] = []
    for index, item in enumerate(raw_data):
        question = str(item.get("question") or item.get("query") or "").strip()
        answers = _flatten_rgb_answers(item.get("answer") or item.get("answers") or [])
        positive_ctxs = _normalize_rgb_contexts(item.get("positive_ctxs") or item.get("positive") or [], prefix=f"positive_{index}")
        negative_ctxs = _normalize_rgb_contexts(item.get("negative_ctxs") or item.get("negative") or [], prefix=f"negative_{index}")
        if question:
            raw_cases.append(
                {
                    "id": item.get("id", index),
                    "question": question,
                    "answers": answers,
                    "positive_ctxs": positive_ctxs,
                    "negative_ctxs": negative_ctxs,
                }
            )
    if case_limit > 0 and len(raw_cases) > case_limit:
        step = len(raw_cases) / case_limit
        raw_cases = [raw_cases[min(len(raw_cases) - 1, int(index * step))] for index in range(case_limit)]

    doc_by_hash: dict[str, CorpusDoc] = {}
    for case in raw_cases:
        for contexts in (case["positive_ctxs"], case["negative_ctxs"]):
            for context in contexts:
                text = clean_text(str(context.get("text") or ""), limit=1600)
                if not text or text == "空白内容":
                    continue
                text_hash = hash_text(text)
                if text_hash in doc_by_hash:
                    continue
                doc_by_hash[text_hash] = CorpusDoc(
                    doc_id=len(doc_by_hash) + 1,
                    title=str(context.get("title") or text_hash).strip() or text_hash,
                    content=text,
                    dataset="rgb",
                )

    cases: list[ExperimentCase] = []
    for case in raw_cases:
        positive_texts = tuple(
            clean_text(str(context.get("text") or ""), limit=1600)
            for context in case["positive_ctxs"]
            if clean_text(str(context.get("text") or ""), limit=1600) != "空白内容"
        )
        expected_doc_ids = tuple(
            doc_by_hash[hash_text(text)].doc_id for text in positive_texts if hash_text(text) in doc_by_hash
        )
        cases.append(
            ExperimentCase(
                dataset="rgb",
                case_id=f"rgb_{case['id']}",
                split="rgb",
                query=case["question"],
                reference="；".join(case["answers"]),
                answers=tuple(case["answers"]),
                positive_texts=positive_texts,
                anchor=case["question"],
                expected_doc_ids=expected_doc_ids,
            )
        )
    return cases, list(doc_by_hash.values())


def _canonical_text(value: str) -> str:
    """把原始文本规整成适合比较与去重的形式。

    这里主要处理 HTML 转义、首尾空白和连续空白折叠，目的是让同一篇新闻
    即便在原数据中存在轻微格式差异，也尽量映射到一致的文本表示。
    """
    return re.sub(r"\s+", " ", html.unescape(str(value or "")).strip())


def _positive_texts_from_items(values: Sequence[str]) -> tuple[str, ...]:
    """从原始字段中提取并清洗正样本文本集合。

    CRUD 的一条样例可能对应 1 到 3 篇支撑新闻，这里统一把它们规整成元组，
    作为后续 expected_doc_ids 映射和命中率分析的基准文本。
    """
    return tuple(_canonical_text(value) for value in values if _canonical_text(value))


def _iter_crud_corpus_texts() -> Iterable[str]:
    """顺序遍历 CRUD 官方语料中的全部新闻正文。

    语料规模较大，不适合一次性全部读入内存。这里做成生成器，
    供后面的正样本匹配和干扰文档抽样按流式方式消费。
    """
    for path in sorted(CRUD_DOCS_ROOT.iterdir()):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = _canonical_text(line)
                if text:
                    yield text


def _evenly_sample(items: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """按均匀间隔从样例列表中抽取固定数量的条目。

    这不是随机抽样，而是更偏展示/复现实验的保守采样方式。
    它可以避免样例过度集中在原列表前部，同时保证每次运行在相同输入下得到一致结果。
    """
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[min(len(items) - 1, int(index * step))] for index in range(limit)]


def _load_crud_split_data() -> dict[str, list[dict[str, Any]]]:
    """读取 CRUD 官方划分文件并做基础格式校验。

    后续正样例构造依赖多个 split，例如 `event_summary`、`questanswer_2docs` 等，
    因此这里先统一把原始 JSON 读成按 split 分组的字典结构，并提前检查数据形态是否正常。
    """
    raw = json.loads(CRUD_SPLIT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("CRUD-RAG 切分数据格式异常。")
    return {str(key): list(value) for key, value in raw.items()}


def _build_crud_positive_cases(
    split_data: dict[str, list[dict[str, Any]]],
    *,
    summary_samples: int,
    qa_1doc_samples: int,
    qa_2doc_samples: int,
    qa_3doc_samples: int,
    hallu_samples: int,
) -> tuple[list[ExperimentCase], dict[str, str]]:
    """构建 CRUD 中需要命中文档才能回答的正样例集合。

    这里会从事件概括、单文档问答、多文档问答和幻觉纠正四类任务中抽样，
    然后统一整理成 `ExperimentCase`。同时还会收集所有正样本文本，
    供后面构建 CRUD 语料子集时保证这些支撑文档一定被纳入语料。
    """
    cases: list[ExperimentCase] = []
    positive_texts_by_hash: dict[str, str] = {}

    def add_case(case: ExperimentCase, source_texts: Sequence[str]) -> None:
        """追加一条样例，并登记其对应的正样本文本。

        内层函数单独存在，是因为每类 split 都要重复做“加入 case + 缓存正样本正文”
        这两步动作，收口后主流程更容易看清不同 split 的构造差异。
        """
        cases.append(case)
        for text in source_texts:
            normalized = _canonical_text(text)
            if not normalized:
                continue
            positive_texts_by_hash[hash_text(normalized)] = normalized

    for index, item in enumerate(_evenly_sample(split_data["event_summary"], summary_samples), start=1):
        source_text = _canonical_text(item.get("text", ""))
        event = str(item.get("event") or "").strip()
        summary = str(item.get("summary") or "").strip()
        add_case(
            ExperimentCase(
                dataset="crud",
                case_id=f"event_summary_{index:03d}",
                split="event_summary",
                query=f"请根据知识库概括这件事的核心内容：{event}",
                reference=summary,
                answers=(summary,),
                positive_texts=_positive_texts_from_items((source_text,)),
                anchor=event or summary,
            ),
            (source_text,),
        )

    for split_name, sample_limit, news_keys in (
        ("questanswer_1doc", qa_1doc_samples, ("news1",)),
        ("questanswer_2docs", qa_2doc_samples, ("news1", "news2")),
        ("questanswer_3docs", qa_3doc_samples, ("news1", "news2", "news3")),
    ):
        for index, item in enumerate(_evenly_sample(split_data[split_name], sample_limit), start=1):
            news_texts = tuple(_canonical_text(item.get(key, "")) for key in news_keys)
            answer = str(item.get("answers") or "").strip()
            question = str(item.get("questions") or "").strip()
            event = str(item.get("event") or "").strip()
            add_case(
                ExperimentCase(
                    dataset="crud",
                    case_id=f"{split_name}_{index:03d}",
                    split=split_name,
                    query=question,
                    reference=answer,
                    answers=(answer,),
                    positive_texts=_positive_texts_from_items(news_texts),
                    anchor=event or question,
                ),
                news_texts,
            )

    for index, item in enumerate(_evenly_sample(split_data["hallu_modified"], hallu_samples), start=1):
        beginning = str(item.get("newsBeginning") or "").strip()
        hallucinated = str(item.get("hallucinatedContinuation") or "").strip()
        corrected = str(item.get("hallucinatedMod") or "").strip()
        full_text = _canonical_text(f"{beginning}\n{item.get('newsRemainder', '')}")
        add_case(
            ExperimentCase(
                dataset="crud",
                case_id=f"hallu_modified_{index:03d}",
                split="hallu_modified",
                query=(
                    "请根据知识库判断并纠正下面这段新闻续写中的错误，只输出纠正后的文本。"
                    f"\n新闻开头：{beginning}\n续写：{hallucinated}"
                ),
                reference=corrected,
                answers=(corrected,),
                positive_texts=_positive_texts_from_items((full_text,)),
                anchor=beginning[:80] or corrected[:80],
            ),
            (full_text,),
        )
    return cases, positive_texts_by_hash


def _build_crud_negative_cases(anchors: Sequence[str], limit: int) -> list[ExperimentCase]:
    """构建 CRUD 的拒答型负样例。

    这些问题会沿用正样例里的事件锚点，但故意询问正文通常不包含的邮箱等细节，
    用于测试系统在证据不足时能否稳定拒答，而不是胡乱生成看似完整的答案。
    """
    negatives: list[ExperimentCase] = []
    for index, item in enumerate(_evenly_sample([{"anchor": value} for value in anchors if value], limit), start=1):
        subject = str(item["anchor"]).strip()
        negatives.append(
            ExperimentCase(
                dataset="crud",
                case_id=f"negative_rejection_{index:03d}",
                split="negative_rejection",
                query=f"关于“{subject}”，文中负责人的电子邮箱地址是什么？如果资料没有提供，请直接说明无法确定。",
                reference="",
                answers=(),
                positive_texts=(),
                should_refuse=True,
                anchor=subject,
            )
        )
    return negatives


def find_crud_equivalent_doc_ids(case_id: str, docs: Sequence[CorpusDoc]) -> tuple[int, ...]:
    """在当前语料中查找与官方标注语义等价的 CRUD 文档。

    这个函数不参与主评测打分，只用于补充分析：某些样例虽然没有命中官方标注 doc_id，
    但可能命中了本地语料里内容等价的文章。答辩或误差分析时，这个信息很有用。
    """
    signatures = CRUD_EQUIVALENT_EVIDENCE_SIGNATURES.get(case_id)
    if not signatures:
        return ()
    matched: list[int] = []
    for doc in docs:
        content = doc.content
        if all(signature in content for signature in signatures):
            matched.append(int(doc.doc_id))
    return tuple(matched)


def _title_from_text(text: str, doc_id: int) -> str:
    """根据正文内容生成一个稳定、可读的伪标题。

    CRUD 语料的原始新闻行文本未必总带明确标题，因此这里会从正文开头裁一段做标题，
    同时拼上 doc_id，保证实验输出和检索来源展示时能快速区分不同文档。
    """
    cleaned = _canonical_text(text)
    cleaned = re.sub(r"^原标题[:：]\s*", "", cleaned)
    cleaned = re.sub(r"^\[\s*\d{4}.*?\]\s*[，,]?", "", cleaned)
    cleaned = cleaned.replace("正文：", " ").strip()
    return f"CRUD_DOC_{doc_id:05d}_{cleaned[:36]}" if cleaned else f"CRUD_DOC_{doc_id:05d}"


def _build_crud_corpus_subset(
    positive_texts_by_hash: dict[str, str],
    *,
    distractor_count: int,
    seed: int,
) -> tuple[list[CorpusDoc], dict[str, int]]:
    """构建论文实验使用的 CRUD 语料子集。

    目标不是复刻官方 8 万篇全文语料，而是在保留全部正样本文档的前提下，
    再按 reservoir sampling 抽取固定数量的干扰文档，形成规模可控、
    但仍具备真实检索难度的子语料。

    返回的 `hash_to_doc_id` 映射会被后面的 `expected_doc_ids` 回填逻辑复用。
    """
    positive_hashes = set(positive_texts_by_hash.keys())
    matched_positives: dict[str, str] = {}
    reservoir: list[str] = []
    rng = random.Random(seed)
    seen_distractors = 0

    for text in _iter_crud_corpus_texts():
        text_hash = hash_text(text)
        if text_hash in positive_hashes and text_hash not in matched_positives:
            matched_positives[text_hash] = text
            continue

        if distractor_count <= 0:
            continue
        seen_distractors += 1
        if len(reservoir) < distractor_count:
            reservoir.append(text)
            continue
        replace_at = rng.randint(0, seen_distractors - 1)
        if replace_at < distractor_count:
            reservoir[replace_at] = text

    docs: list[CorpusDoc] = []
    hash_to_doc_id: dict[str, int] = {}

    def append_doc(text: str) -> None:
        """把一篇文本追加进最终语料，并同步登记哈希到文档编号的映射。"""
        doc_id = len(docs) + 1
        docs.append(CorpusDoc(doc_id=doc_id, title=_title_from_text(text, doc_id), content=text, dataset="crud"))
        hash_to_doc_id[hash_text(text)] = doc_id

    for positive_hash, text in positive_texts_by_hash.items():
        append_doc(matched_positives.get(positive_hash, text))

    for text in reservoir:
        text_hash = hash_text(text)
        if text_hash in hash_to_doc_id:
            continue
        append_doc(text)
    return docs, hash_to_doc_id


def load_crud_cases(
    *,
    summary_samples: int = 10,
    qa_1doc_samples: int = 10,
    qa_2doc_samples: int = 10,
    qa_3doc_samples: int = 10,
    hallu_samples: int = 10,
    negative_samples: int = 16,
    distractor_count: int = 1200,
    seed: int = 42,
) -> tuple[list[ExperimentCase], list[CorpusDoc]]:
    """加载 CRUD 实验样例，并生成对应的检索语料。

    这是 CRUD 数据侧的总入口。函数会先构建正样例和负样例，再生成包含正样本与干扰文档的
    语料子集，最后回填每条样例在当前语料中的 `expected_doc_ids`，供检索命中率、
    作答质量和证伪实验统一使用。
    """
    split_data = _load_crud_split_data()
    positive_cases, positive_texts_by_hash = _build_crud_positive_cases(
        split_data,
        summary_samples=summary_samples,
        qa_1doc_samples=qa_1doc_samples,
        qa_2doc_samples=qa_2doc_samples,
        qa_3doc_samples=qa_3doc_samples,
        hallu_samples=hallu_samples,
    )
    negative_cases = _build_crud_negative_cases([case.anchor for case in positive_cases if case.anchor], negative_samples)
    docs, hash_to_doc_id = _build_crud_corpus_subset(
        positive_texts_by_hash,
        distractor_count=distractor_count,
        seed=seed,
    )

    all_cases = positive_cases + negative_cases
    for case in all_cases:
        case.expected_doc_ids = tuple(
            hash_to_doc_id[hash_text(text)] for text in case.positive_texts if hash_text(text) in hash_to_doc_id
        )
    return all_cases, docs
