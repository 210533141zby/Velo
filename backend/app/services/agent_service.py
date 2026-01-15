"""
=============================================================================
文件: agent_service.py
描述: AI 智能体核心服务

主要功能：
1. 文档索引 (Indexing)：将 Markdown 文档切分并存储到 ChromaDB 向量数据库中，用于后续检索。
2. RAG 问答 (Retrieval-Augmented Generation)：基于用户问题检索相关文档，结合上下文生成回答。

依赖组件:
- langchain: AI 应用开发框架，用于处理文本切分、向量存储和模型调用。
- chroma_db: 向量数据库，用于存储文档内容的语义向量。
- redis: 缓存数据库，用于缓存高频问答结果，提升响应速度。
=============================================================================
"""

import os
import asyncio
import json
import hashlib
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document as LangChainDocument
from openai import APITimeoutError

from app.core.config import settings
from app.logger import logger
from app.services.base import BaseService
from app.cache import redis_manager

# =============================================================================
# 全局配置
# =============================================================================

# 向量数据库的存储位置
# 我们把这些数据存在 backend/chroma_db 文件夹里
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

# 这是一个单例变量，用来存向量数据库的实例
# 简单说，就是为了保证全公司（整个程序）只用这一个数据库连接，别乱开。
_vector_store_instance = None

# 初始化 OpenAI 对话模型实例
# 用于执行文本润色、续写和问答任务
llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_API_BASE,
    model_name="gpt-3.5-turbo",
    temperature=0.3 # 温度参数：0.3 较为保守，适合知识库问答；值越高回答越发散
)

# 初始化 Embeddings 工具
# 用于将文本转换为向量 (Vector)，以便计算文本之间的语义相似度
embeddings = OpenAIEmbeddings(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_API_BASE,
    model=settings.EMBEDDING_MODEL
)

def get_vector_store():
    """
    获取向量数据库实例 (单例模式)
    
    Why:
    确保整个应用程序生命周期内只创建一个 ChromaDB 连接实例，避免资源浪费和潜在的锁冲突。
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        # 如果实例不存在，则创建一个新的持久化 Chroma 向量库
        _vector_store_instance = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    return _vector_store_instance

class AgentService(BaseService):
    """
    AI 服务类
    封装了所有与 AI 模型和向量数据库交互的业务逻辑。
    """
    def __init__(self, db: AsyncSession):
        super().__init__(db)
        # 初始化服务时，确保向量数据库连接已就绪
        self.vector_store = get_vector_store()

    async def index_document(self, doc_id: int, title: str, content: str):
        """
        核心功能：文档索引
        
        将文档内容进行切分、向量化，并存储到向量数据库中，以便后续检索。
        
        Input: 
            doc_id: 文档的唯一标识 ID
            title: 文档标题
            content: 文档正文 (Markdown 格式)
            
        Logic Flow (执行步骤):
            第一步：前置校验。检查 OpenAI API Key 是否配置有效。
            第二步：一级切分 (按章节)。使用 MarkdownHeaderTextSplitter 基于标题 (#) 进行初步切分，保持章节完整性。
            第三步：二级切分 (按长度)。使用 RecursiveCharacterTextSplitter 对过长片段进行细分，适应模型 Context 窗口限制。
            第四步：构建元数据。为每个切分片段添加 source (标题) 和 doc_id (ID) 等标记。
            第五步：向量存储。调用 `vector_store.add_documents` 将处理后的文档片段存入数据库。
            第六步：异常重试。针对网络波动等临时性错误，实施指数退避重试策略。
            
        Why:
            - 为什么要 run_in_threadpool? 
              向量存储涉及磁盘 I/O 和网络请求（调用 Embedding API），属于同步阻塞操作。
              为了避免阻塞 FastAPI 的主事件循环导致服务假死，将其放入线程池执行。
        """
        # 检查是否配置了有效的 API Key
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("sk-placeholder"):
            logger.warning(f"未配置有效的 OpenAI API Key，跳过文档索引 (doc_id={doc_id})")
            return

        if not content:
            return

        try:
            # 第一步：按 Markdown 标题切分
            # 优先保留文档的层级结构，避免在标题和正文之间截断
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(content)

            # 第二步：按字数进一步切分
            # 设定 chunk_size=1000 字符，chunk_overlap=200 字符 (保证上下文连续性)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_splits = text_splitter.split_documents(md_header_splits)
            
            # 兜底逻辑：如果文档没有标准 Markdown 标题，直接按字符切分
            if not final_splits and content:
                final_splits = text_splitter.create_documents([content])

            # 第三步：准备入库数据
            docs = []
            for split in final_splits:
                # 确保 metadata 字典已初始化
                if not split.metadata:
                    split.metadata = {}
                
                # 注入元数据：用于检索时识别文档来源
                split.metadata["source"] = title
                split.metadata["doc_id"] = doc_id
                docs.append(split)
            
            if not docs:
                return

            # 第四步：执行存储 (带重试机制)
            max_retries = 3
            retry_delay = 2
            
            for retry in range(max_retries):
                try:
                    # 将同步的存储操作放入线程池执行
                    await run_in_threadpool(self.vector_store.add_documents, docs)
                    
                    logger.info(f"已索引文档 {doc_id}: {title}", extra={
                        "extra_data": {
                            "event": "rag_index_success",
                            "document_id": doc_id,
                            "chunk_count": len(docs)
                        }
                    })
                    break # 成功则退出重试循环
                    
                except (APITimeoutError, Exception) as e:
                    if retry < max_retries - 1:
                        # 记录警告并等待重试
                        logger.warning(f"索引文档失败 (尝试 {retry + 1}/{max_retries}): {e}，将在 {retry_delay} 秒后重试...", extra={
                            "extra_data": {
                                "event": "rag_index_retry",
                                "document_id": doc_id,
                                "retry": retry + 1
                            }
                        })
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2 # 指数退避：每次重试等待时间翻倍
                    else:
                        raise e # 达到最大重试次数，抛出异常
                        
        except Exception as e:
            logger.error(f"索引文档最终失败: {e}", exc_info=True, extra={
                "extra_data": {
                    "event": "rag_index_failed",
                    "document_id": doc_id
                }
            })

    async def delete_document_index(self, doc_id: int):
        """
        从向量库中删除文档
        
        Logic Flow:
            根据 doc_id 元数据字段，匹配并删除向量数据库中的所有相关片段。
        """
        try:
            # 使用 run_in_threadpool 避免阻塞主线程
            await run_in_threadpool(self.vector_store.delete, where={"doc_id": doc_id})
            
            logger.info(f"已删除文档索引: {doc_id}", extra={
                "extra_data": {
                    "event": "rag_delete_success",
                    "document_id": doc_id
                }
            })
        except Exception as e:
            logger.error(f"删除文档索引失败: {e}", exc_info=True, extra={
                "extra_data": {
                    "event": "rag_delete_failed",
                    "document_id": doc_id
                }
            })

    async def polish_text(self, text: str) -> str:
        """
        Agent 动作: 润色/优化文本
        
        调用 LLM 对输入文本进行专业化润色，提升表达的清晰度和专业性。
        """
        prompt = f"""
        You are a professional editor. Please polish the following text to make it more concise, clear, and professional, while maintaining the original meaning.
        
        Original Text:
        {text}
        
        Polished Text:
        """
        # ainvoke 是 LangChain 的异步调用方法
        response = await llm.ainvoke(prompt)
        
        logger.info("文本润色完成", extra={
            "extra_data": {
                "event": "ai_polish_success",
                "text_length": len(text)
            }
        })
        return response.content

    async def complete_text(self, text: str) -> str:
        """
        Agent 动作: 自动补全文本
        
        调用 LLM 根据当前上下文续写文本。
        """
        prompt = f"""
        You are a helpful writing assistant. Please continue writing the following text naturally.
        
        Context:
        {text}
        
        Continuation:
        """
        response = await llm.ainvoke(prompt)
        
        logger.info("文本续写完成", extra={
            "extra_data": {
                "event": "ai_complete_success",
                "text_length": len(text)
            }
        })
        return response.content

    async def rag_qa(self, query: str) -> dict:
        """
        核心功能：RAG 问答 (检索增强生成)
        
        结合 向量检索 和 LLM 生成，基于知识库内容回答用户问题。支持 Redis 缓存。
        
        Input:
            query: 用户的自然语言提问
        Output:
            dict: { "response": "回答内容", "sources": [参考文档列表] }
            
        Logic Flow (执行步骤):
            第一步：生成 Cache Key。对用户问题进行 MD5 哈希，作为 Redis 缓存的唯一键。
            第二步：查询缓存 (Cache Lookup)。如果 Redis 中存在对应结果，直接返回 (Cache Hit)。
            第三步：向量检索 (Vector Search)。如果缓存未命中，在 ChromaDB 中检索 Top-3 相关文档片段。
            第四步：构建上下文 (Context Construction)。将检索到的片段拼接为背景信息字符串。
            第五步：LLM 生成 (Generation)。将问题和背景信息组合成 Prompt，请求 LLM 生成回答。
            第六步：写入缓存 (Cache Update)。将结果存入 Redis (设置 1 小时过期)，以加速后续相同提问。
            
        Why:
            - 为什么要先查 Redis?
              LLM 推理成本高且延迟大 (通常秒级)。缓存层能显著降低响应延迟 (毫秒级) 并减少 Token 消耗。
        """
        try:
            # 第一步：生成问题的唯一哈希值
            query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
            cache_key = f"rag:response:{query_hash}"
            
            # 第二步：尝试从 Redis 获取缓存结果
            cached_result = await redis_manager.get(cache_key)
            if cached_result:
                # 缓存命中：反序列化并返回
                logger.info(f"[CACHE HIT] RAG 问答命中缓存: {query}", extra={
                    "extra_data": {
                        "event": "rag_cache_hit",
                        "query": query
                    }
                })
                return json.loads(cached_result)

            # --- 以下是缓存未命中 (Cache Miss) 的处理逻辑 ---

            # 第三步：执行向量检索
            # 检索与 query 最相关的 3 个文档片段
            docs = await run_in_threadpool(
                self.vector_store.similarity_search, 
                query, 
                k=3
            )
            
            # 第四步：构建上下文
            context = "\n\n".join([d.page_content for d in docs])
            
            # 整理参考来源信息
            sources = []
            seen_sources = set()
            for d in docs:
                source_name = d.metadata.get("source", "Unknown")
                # 来源去重
                if source_name not in seen_sources:
                    sources.append({
                        "title": source_name,
                        "content": d.page_content[:100] + "..." # 截取前100字作为预览
                    })
                    seen_sources.add(source_name)
            
            # 第五步：调用 LLM 生成回答
            prompt = f"""
            You are a knowledgeable assistant. Answer the question based on the following context. 
            If the answer is not in the context, say "I don't know based on the provided information".
            
            Context:
            {context}
            
            Question: 
            {query}
            
            Answer:
            """
            response = await llm.ainvoke(prompt)
            
            result = {
                "response": response.content,
                "sources": sources
            }
            
            # 第六步：写入 Redis 缓存 (有效期 3600 秒)
            await redis_manager.set(cache_key, json.dumps(result), ex=3600)
            
            logger.info(f"RAG 问答完成: {query}", extra={
                "extra_data": {
                    "event": "rag_qa_success",
                    "query": query,
                    "source_count": len(sources)
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"RAG 问答失败: {e}", exc_info=True, extra={
                "extra_data": {
                    "event": "rag_qa_failed",
                    "query": query
                }
            })
            # 异常处理：返回友好的错误提示，避免前端崩溃
            return {
                "response": "抱歉，系统暂时无法回答您的请求。",
                "sources": []
            }

    async def ask_ai(self, query: str) -> str:
        """
        纯 AI 对话 (非 RAG 模式)
        
        直接调用 LLM 进行通用对话，不依赖本地知识库。
        """
        prompt = f"You are a helpful assistant. Please answer the following question:\n\n{query}"
        response = await llm.ainvoke(prompt)
        return response.content
