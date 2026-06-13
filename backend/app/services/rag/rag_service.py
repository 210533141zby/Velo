from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from typing import Any, Sequence

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import redis_manager
from app.core.config import settings
from app.logger import logger
from app.models import Document
from app.services.model_factory import get_chat_model
from app.services.rag.answer_generators import FallbackRequiredError, GeneratorFactory
from app.services.rag.answer_mode_router import AnswerModeRouter
from app.services.rag.deep_evidence_reviewer import deep_review_and_answer
from app.services.rag.evidence_scorer import UnifiedEvidenceScorer
from app.services.rag.hybrid_search import (
    build_hybrid_candidates,
    ensure_hybrid_index,
    extract_identifiers,
    get_hybrid_index,
    hybrid_index_needs_refresh,
    invalidate_hybrid_index,
    tokenize_for_bm25,
)
from app.services.rag.pipeline_models import AnswerMode, AnswerPlan, EvidenceAssessment, QueryIntent, RetrievedCandidate
from app.services.rag.paper_final_pipeline import (
    NO_CONTEXT_ANSWER as PAPER_FINAL_NO_CONTEXT_ANSWER,
    generate_paper_final_answer,
    paper_final_answer_is_acceptable,
    should_use_complex_paper_path,
)
from app.services.rag.prompt_templates import (
    build_no_context_answer,
    build_simple_rag_prompt,
)
from app.services.rag.query_intent_builder import QueryIntentBuilder
from app.services.rag.source_confidence import (
    calibrated_assessment_confidence,
    calibrated_candidate_confidence,
    rank_assessments_for_confidence,
    rank_candidates_for_confidence,
)
from app.services.rag.vector_index_service import collection_name, delete_document_chunks, get_vector_store, index_document_chunks
from app.services.rag.rerank_service import get_reranker, normalize_lookup_text

RAG_CACHE_VERSION = 'v38'


def _document_key(doc: Any) -> Any:
    """为检索片段生成稳定的来源键。

    这里统一把向量检索、BM25、重排阶段流转的片段对象映射到同一个来源标识，
    这样后面做去重、分数合并和来源展示时，不会因为对象实例不同而把同一篇文档
    误当成多条独立证据。
    """
    metadata = getattr(doc, 'metadata', {}) or {}
    return metadata.get('doc_id') or str(metadata.get('source') or '')


class RagService:
    def __init__(self, db: AsyncSession):
        """初始化 RAG 服务的运行依赖。

        当前对象负责把检索、证据评估、答案路由和结果缓存串成一条后端链路，
        因此这里提前挂好数据库会话、证据评分器、回答路由器以及生成器工厂。
        向量库本身保留为惰性加载，避免应用启动阶段无谓占用资源。
        """
        self.db = db
        self._vector_store = None
        self.evidence_scorer = UnifiedEvidenceScorer()
        self.answer_router = AnswerModeRouter()
        self.generator_factory = GeneratorFactory(get_chat_model)

    @property
    def vector_store(self):
        """惰性获取向量库实例。

        索引对象在问答链路里只在真正发起检索时才需要，因此这里延迟创建。
        这样做可以降低启动阶段的阻塞，并把向量库初始化失败限制在实际调用点。
        """
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    def _cache_key(self, query: str) -> str:
        """根据查询和当前配置生成缓存键。

        缓存键不只依赖原始问题，还会把集合名、索引目录和主链路配置一起纳入签名。
        这样在切换检索模式、候选数量或论文链路版本后，旧缓存会自动失效，
        避免把不同配置下的回答混用到同一个问题上。
        """
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        scope = hashlib.md5(
            json.dumps(
                {
                    'collection': collection_name(),
                    'data_dir': str(settings.data_dir),
                    'pipeline_mode': self._pipeline_mode(),
                    'retrieval_mode': str(settings.RAG_RETRIEVAL_MODE),
                    'vector_limit': int(settings.RAG_VECTOR_SEARCH_LIMIT),
                    'bm25_limit': int(settings.RAG_BM25_SEARCH_LIMIT),
                    'candidate_limit': int(settings.RAG_HYBRID_CANDIDATE_LIMIT),
                    'result_limit': int(settings.RAG_RESULT_LIMIT),
                    'deep_review_top_k': int(settings.RAG_DEEP_REVIEW_TOP_K),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode('utf-8')
        ).hexdigest()
        return f'rag:response:{RAG_CACHE_VERSION}:{scope}:{query_hash}'

    async def _get_cached_result(self, query: str) -> dict[str, Any] | None:
        """读取指定查询的缓存结果。

        命中缓存时直接返回前端需要的完整结构，避免重复执行检索、重排和生成。
        这里也会记录一次缓存命中日志，便于后面排查问答延迟和重复请求行为。
        """
        cached_payload = await redis_manager.get(self._cache_key(query))
        if not cached_payload:
            return None
        logger.info(
            'RAG 问答命中缓存',
            extra={'extra_data': {'event': 'rag_cache_hit', 'query': query}},
        )
        return json.loads(cached_payload)

    async def _cache_result(self, query: str, result: dict[str, Any]) -> dict[str, Any]:
        """写入问答结果缓存并返回结果。

        设计成“写入后原样返回”的形式，是为了让主链路在每个出口都能直接
        `return await self._cache_result(...)`，减少重复样板代码。
        """
        await redis_manager.set(self._cache_key(query), json.dumps(result, ensure_ascii=False), ex=3600)
        return result

    def _pipeline_mode(self) -> str:
        """解析当前启用的 RAG 主链路模式。

        后端保留了简化链路、深度复核链路和论文最终链路几个入口，
        这里负责把配置层的字符串统一归一化，避免业务分支里重复处理大小写和空值。
        """
        return str(settings.RAG_PIPELINE_MODE or 'classic').strip().lower()

    def _sources_from_candidates(self, candidates: Sequence[RetrievedCandidate], used_ranks: Sequence[int]) -> list[dict[str, Any]]:
        """根据候选文档整理前端展示的来源列表。

        deep review 等链路返回的是“用了第几名候选”的信息，而不是最终证据评估结果，
        所以这里要把候选对象重新映射成前端引用列表，并补齐 doc_id、标题和校准后的证据分。
        """
        ranked_candidates = rank_candidates_for_confidence(candidates)
        sources: list[dict[str, Any]] = []
        for output_rank, candidate_rank in enumerate(used_ranks[: settings.RAG_RESULT_LIMIT], start=1):
            index = int(candidate_rank) - 1
            if index < 0 or index >= len(candidates):
                continue
            candidate = candidates[index]
            sources.append(
                {
                    'title': candidate.title,
                    'doc_id': candidate.doc_id,
                    'rank': output_rank,
                    'confidence': calibrated_candidate_confidence(
                        candidate,
                        rank_index=max(ranked_candidates.index(candidate), 0),
                        ranked_candidates=ranked_candidates,
                    ),
                }
            )
        return sources

    def _sources_from_assessments(self, assessments: Sequence[EvidenceAssessment]) -> list[dict[str, Any]]:
        """根据证据评估结果整理前端展示的来源列表。

        论文主链路里真正进入回答阶段的是证据评估后的 assessment，
        因此这里会按最终可用证据去重同源文档，并把校准后的置信度转换成前端可展示的来源分数。
        """
        ranked_assessments = rank_assessments_for_confidence(assessments)
        sources: list[dict[str, Any]] = []
        seen: set[object] = set()
        for rank, assessment in enumerate(assessments, start=1):
            candidate = assessment.candidate
            source_key = candidate.doc_id if candidate.doc_id is not None else candidate.title
            if source_key in seen:
                continue
            seen.add(source_key)
            sources.append(
                {
                    'title': candidate.title,
                    'doc_id': candidate.doc_id,
                    'rank': len(sources) + 1,
                    'confidence': calibrated_assessment_confidence(
                        assessment,
                        rank_index=max(ranked_assessments.index(assessment), 0),
                        ranked_assessments=ranked_assessments,
                    ),
                }
            )
            if len(sources) >= settings.RAG_RESULT_LIMIT:
                break
        return sources

    def _deep_review_candidates(self, candidates: Sequence[RetrievedCandidate]) -> list[RetrievedCandidate]:
        """挑选进入深度复核阶段的高分候选文档。

        深度复核链路成本更高，不会把所有候选都送进模型，而是先按重排分、
        自适应分和覆盖分做一次保守截断，只保留最值得精看的一小批文档。
        """
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(item.rerank_score),
                float(item.adaptive_score),
                float(item.coverage_score),
            ),
            reverse=True,
        )
        return ranked[: settings.RAG_DEEP_REVIEW_TOP_K]

    def _collapse_scored_matches(self, query: str, scored_matches: Sequence[tuple[Any, float]]) -> list[tuple[Any, float]]:
        """按来源合并重复片段，并保留分数最高的结果。

        向量检索常会返回同一篇文档的多个相邻块，如果直接把它们都送到后续融合阶段，
        很容易抬高单篇文档的存在感。这里先按来源折叠，再结合编号命中和词项命中
        做轻量补分，尽量把“真正更贴题的那一块”保留下来。
        """
        query_identifiers = extract_identifiers(query)
        query_tokens = {
            normalize_lookup_text(token)
            for token in tokenize_for_bm25(query)
            if normalize_lookup_text(token)
        }
        best_match_by_source: dict[Any, tuple[Any, float]] = {}
        for doc, score in scored_matches:
            source_key = _document_key(doc)
            content = normalize_lookup_text(getattr(doc, 'page_content', '') or '')
            identifier_hits = sum(1 for identifier in query_identifiers if identifier and identifier in content)
            token_hits = sum(1 for token in query_tokens if token and token in content)
            match_score = float(score) + identifier_hits * 10.0 + token_hits * 0.1
            existing = best_match_by_source.get(source_key)
            if existing is None or match_score > float(existing[1]):
                best_match_by_source[source_key] = (doc, match_score)
        return sorted(best_match_by_source.values(), key=lambda item: item[1], reverse=True)

    async def _load_active_documents(self) -> list[Document]:
        """加载当前启用且正文非空的文档。"""
        result = await self.db.execute(
            select(Document)
            .where(Document.is_active == True)
            .order_by(Document.updated_at.desc().nullslast(), Document.id.desc())
        )
        return [document for document in result.scalars().all() if document.content]

    def _build_retrieved_candidates(
        self,
        candidate_matches: Sequence[tuple[Any, float]],
        rerank_scores: dict[Any, float],
        document_by_id: dict[int, Document],
    ) -> list[RetrievedCandidate]:
        """把检索片段和原始文档整理成统一候选对象。

        检索阶段拿到的是片段级对象，数据库里保存的是文档级正文，而后面的评估和作答
        又同时需要片段文本、全文内容、各路检索分数和来源元数据。这里把这些信息整理成
        `RetrievedCandidate`，让后续模块只面对一种稳定的数据结构。
        """
        retrieved_candidates: list[RetrievedCandidate] = []
        for doc, adaptive_score in candidate_matches:
            metadata = dict(getattr(doc, 'metadata', {}) or {})
            raw_doc_id = metadata.get('doc_id')
            numeric_doc_id = int(raw_doc_id) if raw_doc_id is not None else None
            full_document = document_by_id.get(numeric_doc_id) if numeric_doc_id is not None else None
            full_title = str(getattr(full_document, 'title', '') or metadata.get('source') or '')
            chunk_text = str(metadata.get('raw_text') or getattr(doc, 'page_content', '') or '')
            full_content = str(getattr(full_document, 'content', '') or chunk_text or '')
            resolved_adaptive_score = float(metadata.get('adaptive_score') or adaptive_score or 0.0)
            metadata['adaptive_score'] = resolved_adaptive_score
            candidate_doc = SimpleNamespace(
                page_content=chunk_text,
                chunk_text=chunk_text,
                full_content=full_content,
                metadata={
                    **metadata,
                    'source': full_title,
                    'doc_id': numeric_doc_id,
                },
            )
            retrieved_candidates.append(
                RetrievedCandidate(
                    doc=candidate_doc,
                    doc_id=numeric_doc_id,
                    title=full_title,
                    adaptive_score=resolved_adaptive_score,
                    dense_score=float(metadata.get('vector_score') or 0.0),
                    bm25_score=float(metadata.get('bm25_score') or 0.0),
                    rrf_score=float(metadata.get('rrf_score') or 0.0),
                    rerank_score=float(rerank_scores.get(_document_key(doc), 0.0)),
                    coverage_score=float(metadata.get('coverage_score') or 0.0),
                    identifier_overlap=float(metadata.get('identifier_overlap') or 0.0),
                    chunk_text=chunk_text,
                    full_content=full_content,
                    metadata=metadata,
                )
            )
        return retrieved_candidates

    async def _retrieve_candidates(self, query: str, intent: QueryIntent) -> list[RetrievedCandidate]:
        """执行检索、融合和重排，生成候选文档列表。

        这是问答主链路里“召回侧”的总入口：
        1. 先根据意图决定检索深度和是否使用纠错后的查询；
        2. 执行向量检索并折叠重复块；
        3. 用混合检索把向量结果和 BM25 结果融合；
        4. 调用重排模型得到更稳定的候选顺序；
        5. 最后统一封装成可供证据评分阶段消费的候选对象。
        """
        vector_limit = max(intent.retrieval_depth, settings.RAG_VECTOR_SEARCH_LIMIT)
        bm25_limit = max(intent.retrieval_depth, settings.RAG_BM25_SEARCH_LIMIT)
        candidate_limit = max(intent.retrieval_depth, settings.RAG_HYBRID_CANDIDATE_LIMIT)
        correction_query = 'correction_query' in intent.trace_tags
        retrieval_query = intent.keyword_query if correction_query and intent.keyword_query.strip() else intent.normalized_query

        vector_matches: Sequence[tuple[Any, float]] = []
        try:
            vector_matches = await asyncio.wait_for(
                run_in_threadpool(
                    self.vector_store.similarity_search_with_relevance_scores,
                    retrieval_query,
                    vector_limit,
                ),
                timeout=12.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                '向量检索超时，回退到词法检索候选',
                extra={
                    'extra_data': {
                        'event': 'rag_vector_timeout_fallback',
                        'query': query,
                        'retrieval_query': retrieval_query,
                        'vector_limit': vector_limit,
                    }
                },
            )
        except Exception:
            logger.exception(
                '向量检索失败，回退到词法检索候选',
                extra={
                    'extra_data': {
                        'event': 'rag_vector_error_fallback',
                        'query': query,
                        'retrieval_query': retrieval_query,
                        'vector_limit': vector_limit,
                    }
                },
            )

        collapsed_matches = self._collapse_scored_matches(query, vector_matches) if vector_matches else []
        active_documents = await self._load_active_documents()
        lexical_index = ensure_hybrid_index(active_documents) if hybrid_index_needs_refresh() else get_hybrid_index()
        document_by_id = {int(document.id): document for document in active_documents}
        candidate_matches = build_hybrid_candidates(
            retrieval_query if correction_query else query,
            collapsed_matches,
            lexical_index,
            bm25_query=intent.keyword_query,
            vector_limit=vector_limit,
            bm25_limit=bm25_limit,
            candidate_limit=candidate_limit,
        )
        if not candidate_matches:
            return []

        rerank_scores: dict[Any, float] = {}
        try:
            rerank_scores = await asyncio.wait_for(
                run_in_threadpool(
                    get_reranker().score_documents,
                    retrieval_query if correction_query else query,
                    [doc for doc, _score in candidate_matches],
                ),
                timeout=18.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                'Rerank 超时，保留混合检索候选顺序',
                extra={
                    'extra_data': {
                        'event': 'rag_rerank_timeout_fallback',
                        'query': query,
                        'candidate_count': len(candidate_matches),
                    }
                },
            )
        except Exception:
            logger.exception(
                'Rerank 失败，保留混合检索候选顺序',
                extra={
                    'extra_data': {
                        'event': 'rag_rerank_error_fallback',
                        'query': query,
                        'candidate_count': len(candidate_matches),
                    }
                },
            )
        return self._build_retrieved_candidates(candidate_matches, rerank_scores, document_by_id)

    def _log_intent(self, query: str, intent: QueryIntent) -> None:
        """记录本轮问题的意图识别结果。"""
        logger.info(
            'RAG 意图解析完成',
            extra={
                'extra_data': {
                    'event': 'rag_intent_built',
                    'query': query,
                    'intent_type': intent.intent_type.value,
                    'retrieval_depth': intent.retrieval_depth,
                    'defense_profile': intent.defense_profile.value,
                    'evidence_requirement': intent.evidence_requirement.value,
                    'trace_tags': list(intent.trace_tags),
                }
            },
        )

    def _log_routing(
        self,
        query: str,
        intent: QueryIntent,
        candidates: Sequence[RetrievedCandidate],
        assessments: Sequence[EvidenceAssessment],
        plan: AnswerPlan,
    ) -> None:
        """记录本轮问答的路由决策信息。"""
        logger.info(
            'RAG 路由决策完成',
            extra={
                'extra_data': {
                    'event': 'rag_pipeline_routed',
                    'query': query,
                    'intent_type': intent.intent_type.value,
                    'candidate_count': len(candidates),
                    'assessment_count': len(assessments),
                    'usable_count': sum(1 for assessment in assessments if assessment.usable),
                    'answer_mode': plan.mode.value,
                    'reason': plan.reason,
                    'source_doc_ids': list(plan.source_doc_ids),
                    'route_trace': dict(plan.trace_data),
                }
            },
        )

    def _usable_assessments(self, assessments: Sequence[EvidenceAssessment]) -> list[EvidenceAssessment]:
        """筛出可进入回答阶段的证据评估结果。

        证据评分器会同时给出可用性判断和最终分数。这里统一按最终分降序排序，
        并只保留 `usable=True` 的评估项，供论文链路后面的分项重排、覆盖取证和作答模块使用。
        """
        return [
            assessment
            for assessment in sorted(assessments, key=lambda item: float(item.final_score), reverse=True)
            if assessment.usable
        ]

    def _paper_final_assessments(self, assessments: Sequence[EvidenceAssessment]) -> list[EvidenceAssessment]:
        """为论文主链路提供一组更宽松的候选证据。

        答辩演示时更重要的是“基于现有证据给出可核对回答”，而不是因为阈值过严直接拒答。
        因此当严格 usable 为空时，这里会回退到最终分最高的一小批评估项，让论文主链路
        继续做分项重排和覆盖取证。
        """
        usable = self._usable_assessments(assessments)
        if usable:
            return usable
        return sorted(assessments, key=lambda item: float(item.final_score), reverse=True)[
            : max(int(settings.RAG_PAPER_FINAL_COMPLEX_SOURCE_COUNT), 4)
        ]

    async def _execute_with_fallback(
        self,
        query: str,
        intent: QueryIntent,
        usable_assessments: Sequence[EvidenceAssessment],
        plan: AnswerPlan,
    ) -> tuple[str, list[dict]]:
        """执行回答计划，并在必要时做优雅降级。

        生成器可能因为证据不足、模式不适配等原因主动抛出降级信号。
        这里负责把“复杂模式失败后退到简单模式”的逻辑收口到一处，
        避免不同问答分支各自实现一套回退策略。
        """
        current_plan = plan
        while True:
            try:
                return await self.generator_factory.execute(current_plan, query, intent, usable_assessments)
            except FallbackRequiredError as exc:
                logger.info(
                    'RAG 生成器触发优雅降级',
                    extra={
                        'extra_data': {
                            'event': 'rag_generator_fallback',
                            'query': query,
                            'from_mode': current_plan.mode.value,
                            'reason': str(exc),
                        }
                    },
                )
                current_plan = self.generator_factory.downgrade(current_plan)
                if current_plan.mode is AnswerMode.NO_CONTEXT:
                    return build_no_context_answer(), []

    async def ensure_bootstrap_index(self):
        """确保系统启动后基础索引可用。

        如果数据库里已有文档、但向量集合还是空的，说明当前实例尚未完成首轮建索引。
        这里会在启动期补做一次全量索引预热，避免用户第一次提问时才触发整批文档入库。
        """
        active_docs = await self._load_active_documents()
        if not active_docs:
            return

        collection_count = await run_in_threadpool(self.vector_store._collection.count)
        if collection_count > 0:
            return

        logger.info(
            f'RAG collection 为空，开始预热索引 {len(active_docs)} 篇文档',
            extra={
                'extra_data': {
                    'event': 'rag_bootstrap_start',
                    'document_count': len(active_docs),
                    'collection_name': collection_name(),
                }
            },
        )
        for doc in active_docs:
            await self.index_document(doc.id, doc.title, doc.content)
        logger.info(
            'RAG 预热索引完成',
            extra={
                'extra_data': {
                    'event': 'rag_bootstrap_complete',
                    'document_count': len(active_docs),
                    'collection_name': collection_name(),
                }
            },
        )

    async def index_document(self, doc_id: int, title: str, content: str):
        """为单篇文档建立或刷新索引。

        文档编辑完成后，除了更新向量块，还需要同步失效混合检索侧的词法索引缓存，
        否则问答链路可能继续使用旧的 BM25 视图。
        """
        await index_document_chunks(doc_id, title, content)
        invalidate_hybrid_index()

    async def delete_document_index(self, doc_id: int):
        """删除单篇文档对应的索引。

        删除动作同样要联动失效混合检索缓存，保证候选集合不会再包含已移除的文档内容。
        """
        await delete_document_chunks(doc_id)
        invalidate_hybrid_index()

    async def _run_simple_pipeline(self, query: str, candidates: Sequence[RetrievedCandidate]) -> dict[str, Any]:
        """运行简化版问答链路。

        这条链路只做轻量截断和单轮直接作答，不走证据评分与复杂回答组织。
        它主要用于开发调试、性能兜底或对比实验，帮助确认问题究竟出在检索侧还是论文增强链路。
        """
        top_candidates = sorted(
            candidates,
            key=lambda item: (float(item.rerank_score), float(item.adaptive_score)),
            reverse=True,
        )[: settings.RAG_RESULT_LIMIT]
        if not top_candidates:
            return {'response': build_no_context_answer(), 'sources': []}

        char_budget = max(400, int(settings.RAG_SIMPLE_CONTEXT_CHARS))
        per_source_budget = max(200, char_budget // max(len(top_candidates), 1))
        blocks: list[str] = []
        for index, candidate in enumerate(top_candidates, start=1):
            body = (candidate.chunk_text or candidate.full_content or '').strip()
            if not body:
                body = str(getattr(candidate.doc, 'page_content', '') or '').strip()
            if not body:
                continue
            blocks.append(f"[{index}] {candidate.title}\n{body[:per_source_budget]}")
        if not blocks:
            return {'response': build_no_context_answer(), 'sources': []}

        prompt = build_simple_rag_prompt(query, '\n\n'.join(blocks))
        fallback_answer = '；'.join(block.split('\n', 1)[1].strip() for block in blocks[:2] if '\n' in block).strip()
        try:
            response = await asyncio.wait_for(get_chat_model().ainvoke(prompt), timeout=20.0)
            answer = str(getattr(response, 'content', '') or '').strip()
        except asyncio.TimeoutError:
            logger.warning(
                'Simple RAG 生成超时，回退到证据式答案',
                extra={'extra_data': {'event': 'rag_simple_timeout', 'query': query}},
            )
            answer = fallback_answer
        except Exception:
            logger.exception(
                'Simple RAG 生成失败，回退到证据式答案',
                extra={'extra_data': {'event': 'rag_simple_failed', 'query': query}},
            )
            answer = fallback_answer
        if not answer:
            answer = fallback_answer or build_no_context_answer()

        sources = [
            {
                'title': candidate.title,
                'doc_id': candidate.doc_id,
                'rank': rank,
                'confidence': calibrated_candidate_confidence(
                    candidate,
                    rank_index=rank - 1,
                    ranked_candidates=top_candidates,
                ),
            }
            for rank, candidate in enumerate(top_candidates, start=1)
        ]
        return {'response': answer, 'sources': sources}

    async def rag_qa(self, query: str) -> dict:
        """执行一次完整的 RAG 问答。

        这是后端知识库问答的总入口。整体流程按顺序包括：
        1. 读取缓存；
        2. 解析查询意图；
        3. 完成混合检索与重排；
        4. 按配置选择 simple、deep_review 或论文主链路；
        5. 在论文主链路中执行证据评估、复杂路由和最终作答；
        6. 统一回填来源并写入缓存。

        由于前端所有问答请求最终都会经过这里，所以这里的注释尽量写清
        “总入口如何分流到不同算法模式”，便于后续排查线上回答行为。
        """
        try:
            cached_result = await self._get_cached_result(query)
            if cached_result is not None:
                return cached_result

            intent = await QueryIntentBuilder.build(query)
            self._log_intent(query, intent)
            candidates = await self._retrieve_candidates(query, intent)
            pipeline_mode = self._pipeline_mode()
            if pipeline_mode == 'simple':
                result = await self._run_simple_pipeline(query, candidates)
                return await self._cache_result(query, result)
            if pipeline_mode == 'deep_review':
                review_candidates = self._deep_review_candidates(candidates)
                review = await deep_review_and_answer(query, review_candidates)
                if review.get('verdict') == 'answer' and review.get('answer'):
                    result = {
                        'response': str(review.get('answer') or ''),
                        'sources': self._sources_from_candidates(review_candidates, review.get('used_ranks') or ()),
                    }
                else:
                    result = {'response': build_no_context_answer(), 'sources': []}
                return await self._cache_result(query, result)
            assessments = await self.evidence_scorer.assess_concurrently(candidates, intent)
            usable_assessments = self._usable_assessments(assessments)
            paper_assessments = self._paper_final_assessments(assessments)
            if pipeline_mode == 'paper_final' and paper_assessments:
                complex_path_assessments = usable_assessments or paper_assessments
                if should_use_complex_paper_path(query, intent, complex_path_assessments):
                    paper_result = await generate_paper_final_answer(
                        get_chat_model,
                        query,
                        intent,
                        paper_assessments,
                        final_source_count=int(settings.RAG_PAPER_FINAL_SOURCE_COUNT),
                        complex_source_count=int(settings.RAG_PAPER_FINAL_COMPLEX_SOURCE_COUNT),
                    )
                    if paper_result.answer and (
                        paper_final_answer_is_acceptable(query, intent, paper_result.answer)
                        or not usable_assessments
                    ):
                        result = {
                            'response': paper_result.answer,
                            'sources': self._sources_from_assessments(paper_result.selected_assessments),
                        }
                        await self._cache_result(query, result)
                        logger.info(
                            'RAG 论文方案链路完成',
                            extra={
                                'extra_data': {
                                'event': 'rag_paper_final_success',
                                    'query': query,
                                    'source_count': len(result['sources']),
                                    'candidate_count': len(candidates),
                                    'relaxed_mode': not bool(usable_assessments),
                                    'collection_name': collection_name(),
                                }
                            },
                        )
                        return result
                    logger.info(
                        'RAG 论文方案链路回退到常规生成',
                        extra={
                            'extra_data': {
                                'event': 'rag_paper_final_fallback',
                                'query': query,
                                'raw_answer': str(paper_result.answer or ''),
                                'source_count': len(paper_result.selected_assessments),
                                'collection_name': collection_name(),
                            }
                        },
                    )
            plan = self.answer_router.route(intent, usable_assessments)
            self._log_routing(query, intent, candidates, assessments, plan)
            response, sources = await self._execute_with_fallback(query, intent, usable_assessments, plan)
            result = {'response': response, 'sources': sources}
            await self._cache_result(query, result)
            logger.info(
                'RAG 问答完成',
                extra={
                    'extra_data': {
                        'event': 'rag_qa_success',
                        'query': query,
                        'source_count': len(sources),
                        'candidate_count': len(candidates),
                        'collection_name': collection_name(),
                    }
                },
            )
            return result
        except Exception:
            logger.exception(
                'RAG 问答失败',
                extra={'extra_data': {'event': 'rag_qa_failed', 'query': query, 'collection_name': collection_name()}},
            )
            return {'response': '抱歉，系统暂时无法回答您的请求。', 'sources': []}
