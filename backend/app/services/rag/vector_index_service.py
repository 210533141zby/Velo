from __future__ import annotations

import asyncio
import re

from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.config import settings
from app.logger import logger
from app.services.rag.hybrid_search import invalidate_hybrid_index
from app.services.model_factory import get_rag_contextual_model

_embeddings_instance = None
_vector_store_instance = None
_CONTEXTUAL_PREFIX_CHAR_LIMIT = 240
_DOCUMENT_CONTEXT_CHAR_LIMIT = 2400
_CHUNK_CONTEXT_CHAR_LIMIT = 1200
_CONTEXTUAL_INDEX_CONCURRENCY = 4


def _sanitize_collection_part(value: str) -> str:
    """清洗集合名片段，避免模型名或提供方字符串污染 Chroma 集合名。"""
    cleaned = re.sub(r'[^a-zA-Z0-9_-]+', '_', value or '')
    return cleaned.strip('_').lower() or 'default'


def collection_name() -> str:
    """按嵌入提供方、模型和上下文化开关生成稳定集合名。

    这样不同 embedding 配置可以并存，不会把旧索引和新索引写到同一集合里。
    """
    provider = _sanitize_collection_part(settings.EMBEDDING_PROVIDER)
    model = _sanitize_collection_part(settings.EMBEDDING_MODEL)
    variant = 'ctx' if settings.RAG_CONTEXTUAL_EMBED else 'base'
    return f'velo_{provider}_{model}_{variant}'[:63]


class CpuEmbeddingClient:
    def __init__(self, model_name: str):
        """加载基于 `sentence-transformers` 的本地 CPU 向量模型。"""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device='cpu')

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为一批文本生成归一化向量，供建库使用。"""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """为单条查询生成归一化向量，供相似度检索使用。"""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def get_embeddings():
    """返回当前系统使用的 embedding 客户端单例。

    根据配置自动在本地 HuggingFace 模型和 OpenAI 兼容接口之间切换，
    上层索引逻辑无需感知具体来源。
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        provider = (settings.EMBEDDING_PROVIDER or '').lower()
        _embeddings_instance = (
            CpuEmbeddingClient(settings.EMBEDDING_MODEL)
            if provider == 'huggingface'
            else OpenAIEmbeddings(
                api_key=settings.llm_api_key,
                base_url=settings.embedding_api_base,
                model=settings.EMBEDDING_MODEL,
                check_embedding_ctx_length=False,
            )
        )
    return _embeddings_instance


def get_vector_store():
    """返回 Chroma 向量库单例，并在首次访问时完成初始化。"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = Chroma(
            collection_name=collection_name(),
            persist_directory=str(settings.chroma_persist_directory),
            embedding_function=get_embeddings(),
        )
    return _vector_store_instance


def _compact_text(text: str, limit: int | None = None) -> str:
    """压缩空白并按需截断文本，供切分与上下文化提示复用。"""
    normalized = re.sub(r'\s+', ' ', str(text or '').strip())
    if limit is None:
        return normalized
    return normalized[:limit].strip()


def _section_path(metadata: dict[str, str] | None) -> str:
    """根据 Markdown 标题层级拼出当前片段所在的章节路径。"""
    if not metadata:
        return ''
    sections = [str(metadata.get(key) or '').strip() for key in ('Header 1', 'Header 2', 'Header 3')]
    return ' > '.join(section for section in sections if section)


def _build_contextualized_chunk(prefix: str, raw_text: str) -> str:
    """把上下文化前缀和原始切片拼成最终入库文本。"""
    prefix_text = _compact_text(prefix, _CONTEXTUAL_PREFIX_CHAR_LIMIT)
    chunk_text = _compact_text(raw_text)
    if not prefix_text:
        return chunk_text
    return f'{prefix_text}\n\n{chunk_text}'.strip()


def _build_document_context(title: str, content: str) -> str:
    """为上下文化改写准备一段文档级概览文本。"""
    title_line = _compact_text(title)
    body = _compact_text(content, _DOCUMENT_CONTEXT_CHAR_LIMIT)
    if title_line and body:
        return f'Title: {title_line}\nDocument: {body}'
    return title_line or body


def _build_contextual_prompt(title: str, document_context: str, section_path: str, chunk_text: str) -> str:
    """构建 contextual retrieval 使用的片段补背景提示词。"""
    section_hint = section_path or 'N/A'
    chunk_preview = _compact_text(chunk_text, _CHUNK_CONTEXT_CHAR_LIMIT)
    return (
        'You add minimal document-level context to a chunk for retrieval.\n'
        'Write one or two short factual sentences that place the chunk within the broader document.\n'
        'Do not answer a user question. Do not invent facts. Do not repeat the chunk verbatim. Return plain text only.\n\n'
        f'Document title: {title}\n'
        f'Document overview:\n{document_context}\n\n'
        f'Section path: {section_hint}\n'
        f'Chunk:\n{chunk_preview}\n\n'
        'Context:'
    )


async def _generate_contextual_prefix(title: str, document_context: str, section_path: str, raw_text: str) -> str:
    """调用模型为切片生成最小必要的文档级背景前缀。"""
    prompt = _build_contextual_prompt(title, document_context, section_path, raw_text)
    response = await get_rag_contextual_model().ainvoke(prompt)
    return _compact_text(getattr(response, 'content', ''), _CONTEXTUAL_PREFIX_CHAR_LIMIT)


def _fallback_contextual_prefix(title: str, section_path: str) -> str:
    """当模型改写失败时，用标题和章节路径拼一个低成本兜底前缀。"""
    title_line = _compact_text(title)
    if title_line and section_path:
        return f'{title_line} / {section_path}'
    return title_line or section_path


async def _prepare_documents_for_indexing(doc_id: int, title: str, content: str, splits: list) -> list:
    """把切分结果整理成真正可入库的文档片段列表。

    如果未启用上下文化，就直接写入原始切片；否则会为每个切片补充前缀，
    再把原文和前缀一起送入向量库。
    """
    docs = []
    document_context = _build_document_context(title, content)

    if not settings.RAG_CONTEXTUAL_EMBED:
        for split in splits:
            raw_text = _compact_text(getattr(split, 'page_content', ''))
            if not raw_text:
                continue
            if not split.metadata:
                split.metadata = {}
            split.metadata['source'] = title
            split.metadata['doc_id'] = doc_id
            split.metadata['raw_text'] = raw_text
            split.page_content = raw_text
            docs.append(split)
        return docs

    semaphore = asyncio.Semaphore(_CONTEXTUAL_INDEX_CONCURRENCY)

    async def prepare_split(split):
        """处理单个切片的元数据、原文保存和上下文化前缀生成。"""
        raw_text = _compact_text(getattr(split, 'page_content', ''))
        if not raw_text:
            return None
        if not split.metadata:
            split.metadata = {}
        split.metadata['source'] = title
        split.metadata['doc_id'] = doc_id
        split.metadata['raw_text'] = raw_text
        section_path = _section_path(split.metadata)
        try:
            async with semaphore:
                prefix = await _generate_contextual_prefix(title, document_context, section_path, raw_text)
        except Exception:
            logger.exception(
                '生成 contextual retrieval 前缀失败',
                extra={'extra_data': {'event': 'rag_contextual_prefix_failed', 'document_id': doc_id}},
            )
            prefix = ''
        prefix = prefix or _fallback_contextual_prefix(title, section_path)
        split.metadata['contextual_prefix'] = prefix
        split.page_content = _build_contextualized_chunk(prefix, raw_text)
        return split

    prepared = await asyncio.gather(*(prepare_split(split) for split in splits))
    return [split for split in prepared if split is not None]


async def index_document_chunks(doc_id: int, title: str, content: str):
    """把一篇文档切分后写入向量库，并在成功后作废词法索引缓存。

    这一步既服务语义检索，也为后续混合检索提供和正文同步的最新语料。
    """
    if not content:
        return
    try:
        headers_to_split_on = [('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3')]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_splits = text_splitter.split_documents(md_header_splits)
        if not final_splits and content:
            final_splits = text_splitter.create_documents([content])

        docs = await _prepare_documents_for_indexing(doc_id, title, content, list(final_splits))
        if not docs:
            return

        retry_delay = 2
        vector_store = get_vector_store()
        for retry in range(3):
            try:
                await run_in_threadpool(vector_store.delete, where={'doc_id': doc_id})
                await run_in_threadpool(vector_store.add_documents, docs)
                logger.info(
                    f'已索引文档 {doc_id}: {title}',
                    extra={
                        'extra_data': {
                            'event': 'rag_index_success',
                            'document_id': doc_id,
                            'chunk_count': len(docs),
                            'collection_name': collection_name(),
                        }
                    },
                )
                invalidate_hybrid_index()
                break
            except Exception as exc:
                if retry < 2:
                    logger.warning(
                        f'索引文档失败 (尝试 {retry + 1}/3): {exc}，将在 {retry_delay} 秒后重试...',
                        extra={
                            'extra_data': {
                                'event': 'rag_index_retry',
                                'document_id': doc_id,
                                'retry': retry + 1,
                                'collection_name': collection_name(),
                            }
                        },
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
    except Exception:
        logger.exception(
            '索引文档最终失败',
            extra={
                'extra_data': {
                    'event': 'rag_index_failed',
                    'document_id': doc_id,
                    'collection_name': collection_name(),
                }
            },
        )


async def delete_document_chunks(doc_id: int):
    """删除指定文档在向量库中的全部切片，并通知词法索引失效。"""
    try:
        await run_in_threadpool(get_vector_store().delete, where={'doc_id': doc_id})
        invalidate_hybrid_index()
        logger.info(
            f'已删除文档索引: {doc_id}',
            extra={
                'extra_data': {
                    'event': 'rag_delete_success',
                    'document_id': doc_id,
                    'collection_name': collection_name(),
                }
            },
        )
    except Exception:
        logger.exception(
            '删除文档索引失败',
            extra={
                'extra_data': {
                    'event': 'rag_delete_failed',
                    'document_id': doc_id,
                    'collection_name': collection_name(),
                }
            },
        )
