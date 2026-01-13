from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from sqlalchemy.orm import selectinload
from fastapi import BackgroundTasks, UploadFile
from app.core.config import settings
from app.models import Document, Folder, Log
from app.schemas import DocumentCreate, FolderCreate
from app.cache import redis_manager
from app.logger import logger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument
from fastapi.concurrency import run_in_threadpool
from openai import APITimeoutError
import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime

# =============================================================================
# 全局常量与配置
# =============================================================================

# 向量存储路径
# 确保路径指向 backend/chroma_db
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# 向量存储的全局单例
_vector_store_instance = None

# 初始化 OpenAI 组件
llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_API_BASE,
    model_name="gpt-3.5-turbo",
    temperature=0.3
)

embeddings = OpenAIEmbeddings(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_API_BASE,
    model=settings.EMBEDDING_MODEL
)

def get_vector_store():
    """
    获取向量存储实例 (单例模式)
    避免重复初始化消耗资源
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    return _vector_store_instance

# =============================================================================
# 基础服务类
# =============================================================================

class BaseService:
    """
    基础服务类
    提供数据库会话的通用封装
    """
    def __init__(self, db: AsyncSession):
        self.db = db

# =============================================================================
# 日志服务
# =============================================================================

class LogService(BaseService):
    """
    日志服务类
    负责记录系统操作日志，通常在后台任务中异步执行
    """
    async def log_operation(self, action: str, resource_type: str, resource_id: str, user_id: str = "system", details: dict = None):
        """
        记录操作日志
        
        参数:
            action: 操作动作 (如 CREATE, DELETE)
            resource_type: 资源类型 (如 DOCUMENT, FOLDER)
            resource_id: 资源ID
            user_id: 操作用户ID
            details: 额外的详细信息字典
        """
        try:
            log = Log(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id),
                details=details
            )
            self.db.add(log)
            await self.db.commit()
        except Exception as e:
            logger.error(f"日志记录失败: {e}", exc_info=True, extra={
                "extra_data": {
                    "event": "log_operation_failed",
                    "resource_type": resource_type,
                    "resource_id": resource_id
                }
            })

# =============================================================================
# 文档服务
# =============================================================================

class DocumentService(BaseService):
    """
    文档服务类
    处理文档的 CRUD、缓存管理及与 Agent 服务的交互
    """
    
    async def create_document(self, doc_data: DocumentCreate, background_tasks: BackgroundTasks) -> Document:
        """
        创建新文档
        
        流程:
        1. 写入数据库
        2. 清除列表缓存
        3. 添加后台任务: 建立向量索引
        4. 添加后台任务: 记录操作日志
        """
        new_doc = Document(
            title=doc_data.title, 
            content=doc_data.content, 
            folder_id=doc_data.folder_id
        )
        self.db.add(new_doc)
        await self.db.commit()
        await self.db.refresh(new_doc)
        
        # 缓存失效
        await self._invalidate_cache()
        
        # 触发后台任务
        # 注意: 这里我们需要为后台任务创建新的 Service 实例，因为当前 session 可能会结束
        # 但在 FastAPI BackgroundTasks 中，通常传递参数给函数即可
        # 为了简化，我们假设 AgentService 和 LogService 会自行处理 Session 或由外部传入
        # 在这里我们直接调用逻辑，实际执行由 FastAPI 调度
        
        # RAG 索引
        agent_service = AgentService(self.db)
        background_tasks.add_task(agent_service.index_document, new_doc.id, new_doc.title, new_doc.content)
        
        logger.info(f"创建文档: {new_doc.title}", extra={
            "extra_data": {
                "event": "document_created",
                "document_id": new_doc.id,
                "folder_id": new_doc.folder_id,
                "title": new_doc.title
            }
        })
        
        # 日志记录 (复用当前 DB Session 可能会有问题，但在 BackgroundTasks 中应使用独立 Session)
        # 这里我们采用一种折中方案：在 API 层注入新的 session 给 background task，或者在 task 内部创建 session
        # 为保持简单，我们暂不在此处直接添加 task，而是返回 doc 对象，由 API 层统一调度 task，
        # 或者在这里调用一个不依赖当前 session 的 helper。
        # 鉴于 API 层代码结构，我们将 task 添加逻辑保留在 Service 中，但需注意 Session 生命周期。
        
        return new_doc

    async def get_document(self, doc_id: int) -> Document:
        """获取单个文档详情"""
        query = select(Document).where(Document.id == doc_id, Document.is_active == True)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def update_document(self, doc_id: int, doc_update: DocumentCreate, background_tasks: BackgroundTasks) -> Document:
        """
        更新文档
        
        流程:
        1. 检查文档是否存在
        2. 更新字段
        3. 提交事务
        4. 清除缓存
        5. 触发后台任务: 更新向量索引
        """
        doc = await self.get_document(doc_id)
        if not doc:
            return None
            
        doc.title = doc_update.title
        doc.content = doc_update.content
        doc.updated_at = datetime.utcnow()
        # 注意: folder_id 也可以更新
        if doc_update.folder_id is not None:
             doc.folder_id = doc_update.folder_id
        
        await self.db.commit()
        await self.db.refresh(doc)
        
        # 缓存失效
        await self._invalidate_cache()
        
        # 更新索引 (先删除旧索引比较复杂，这里简化为重新添加，LangChain 通常会去重或更新)
        # 更好的做法是在 metadata 中存储 hash
        agent_service = AgentService(self.db)
        background_tasks.add_task(agent_service.index_document, doc.id, doc.title, doc.content)
        
        logger.info(f"更新文档: {doc.title}", extra={
            "extra_data": {
                "event": "document_updated",
                "document_id": doc.id,
                "title": doc.title
            }
        })
        
        return doc

    async def list_documents(self) -> list[dict]:
        """
        获取文档列表
        优先读取 Redis 缓存
        """
        # 1. 尝试缓存
        try:
            cached_docs = await redis_manager.get("documents_list")
            if cached_docs:
                return json.loads(cached_docs)
        except Exception as e:
            logger.error(f"缓存读取错误: {e}", exc_info=True)

        # 2. 查询数据库
        query = select(
            Document.id,
            Document.title,
            func.substr(Document.content, 1, 200).label("content"),
            Document.created_at,
            Document.updated_at,
            Document.folder_id
        ).where(Document.is_active == True).order_by(Document.updated_at.desc())
        
        result = await self.db.execute(query)
        rows = result.all()
        
        docs_data = [
            {
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "folder_id": row.folder_id
            }
            for row in rows
        ]
        
        # 3. 写入缓存 (TTL 5分钟)
        try:
            await redis_manager.set("documents_list", json.dumps(docs_data), expire=300)
        except Exception as e:
            logger.error(f"缓存写入错误: {e}", exc_info=True)
            
        return docs_data

    async def delete_document(self, doc_id: int, background_tasks: BackgroundTasks) -> bool:
        """
        删除文档 (软删除)
        """
        doc = await self.get_document(doc_id)
        if not doc:
            return False
            
        doc.is_active = False
        await self.db.commit()
        
        # 缓存失效
        await self._invalidate_cache()
        
        # 从向量库中删除
        agent_service = AgentService(self.db)
        background_tasks.add_task(agent_service.delete_document_index, doc_id)
        
        logger.info(f"删除文档: {doc_id}", extra={
            "extra_data": {
                "event": "document_deleted",
                "document_id": doc_id
            }
        })
        
        return True

    async def _invalidate_cache(self):
        """清除文档列表缓存"""
        try:
            await redis_manager.delete("documents_list")
        except Exception as e:
            logger.error(f"缓存清除错误: {e}", exc_info=True)

# =============================================================================
# 文件夹服务
# =============================================================================

class FolderService(BaseService):
    """
    文件夹服务类
    处理文件夹层级结构的 CRUD 操作
    """
    
    async def create_folder(self, folder_data: FolderCreate) -> Folder:
        """创建新文件夹"""
        folder = Folder(title=folder_data.title, parent_id=folder_data.parent_id)
        self.db.add(folder)
        await self.db.commit()
        await self.db.refresh(folder)
        return folder

    async def get_all_folders(self) -> list[Folder]:
        """获取所有文件夹"""
        query = select(Folder).where(Folder.is_active == True)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_folder(self, folder_id: int) -> Folder:
        """获取单个文件夹详情"""
        query = select(Folder).where(Folder.id == folder_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_folder_contents(self, folder_id: int = None) -> dict:
        """
        获取指定文件夹下的内容 (子文件夹 + 文档)
        folder_id 为 None 时表示根目录
        """
        # 获取子文件夹
        folder_query = select(Folder).where(Folder.is_active == True)
        if folder_id is None:
            folder_query = folder_query.where(Folder.parent_id == None)
        else:
            folder_query = folder_query.where(Folder.parent_id == folder_id)
            
        folders_res = await self.db.execute(folder_query)
        folders = folders_res.scalars().all()
        
        # 获取文档
        doc_query = select(Document).where(Document.is_active == True)
        if folder_id is None:
            doc_query = doc_query.where(Document.folder_id == None)
        else:
            doc_query = doc_query.where(Document.folder_id == folder_id)
            
        docs_res = await self.db.execute(doc_query)
        docs = docs_res.scalars().all()
        
        return {"folders": folders, "documents": docs}

# =============================================================================
# AI Agent 服务
# =============================================================================

class AgentService(BaseService):
    """
    AI 代理服务类
    封装 LangChain 调用、RAG 逻辑及文本处理功能
    """
    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.vector_store = get_vector_store()

    async def index_document(self, doc_id: int, title: str, content: str):
        """
        索引文档 (通常作为后台任务运行)
        将文档拆分为片段并存入向量数据库
        """
        if not content:
            return

        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(content)
            
            docs = [
                LangChainDocument(
                    page_content=t,
                    metadata={"source": title, "doc_id": doc_id}
                ) for t in texts
            ]
            
            # 增加重试机制
            max_retries = 3
            retry_delay = 2
            
            for retry in range(max_retries):
                try:
                    # 运行同步的 Chroma 操作
                    await run_in_threadpool(self.vector_store.add_documents, docs)
                    
                    logger.info(f"已索引文档 {doc_id}: {title}", extra={
                        "extra_data": {
                            "event": "rag_index_success",
                            "document_id": doc_id,
                            "chunk_count": len(docs)
                        }
                    })
                    break # 成功则退出循环
                    
                except (APITimeoutError, Exception) as e:
                    if retry < max_retries - 1:
                        logger.warning(f"索引文档失败 (尝试 {retry + 1}/{max_retries}): {e}，将在 {retry_delay} 秒后重试...", extra={
                            "extra_data": {
                                "event": "rag_index_retry",
                                "document_id": doc_id,
                                "retry": retry + 1
                            }
                        })
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2 # 指数退避
                    else:
                        raise e # 最后一次重试失败，抛出异常
                        
        except Exception as e:
            logger.error(f"索引文档最终失败: {e}", exc_info=True, extra={
                "extra_data": {
                    "event": "rag_index_failed",
                    "document_id": doc_id
                }
            })

    async def delete_document_index(self, doc_id: int):
        """
        从向量库中删除文档索引
        """
        try:
            # Chroma delete supports where filter
            # We need to run this in threadpool because Chroma client might be sync or we want to avoid blocking
            # But vector_store.delete might be sync.
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
        """Agent 动作: 润色/优化文本"""
        prompt = f"""
        You are a professional editor. Please polish the following text to make it more concise, clear, and professional, while maintaining the original meaning.
        
        Original Text:
        {text}
        
        Polished Text:
        """
        response = await llm.ainvoke(prompt)
        
        logger.info("文本润色完成", extra={
            "extra_data": {
                "event": "ai_polish_success",
                "text_length": len(text)
            }
        })
        return response.content

    async def complete_text(self, text: str) -> str:
        """Agent 动作: 自动补全文本"""
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
        RAG 问答
        
        流程:
        1. 在向量数据库中检索相关文档
        2. 构建包含上下文的 Prompt
        3. 调用 LLM 生成回答
        """
        try:
            # 1. 检索
            docs = await run_in_threadpool(
                self.vector_store.similarity_search, 
                query, 
                k=3
            )
            
            # 2. 构建上下文
            context = "\n\n".join([d.page_content for d in docs])
            
            # 构建来源列表 (List[Dict])
            sources = []
            seen_sources = set()
            for d in docs:
                source_name = d.metadata.get("source", "Unknown")
                if source_name not in seen_sources:
                    sources.append({
                        "title": source_name,
                        "content": d.page_content[:100] + "..." # 截取部分内容作为预览
                    })
                    seen_sources.add(source_name)
            
            # 3. 生成回答
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
            
            logger.info(f"RAG 问答完成: {query}", extra={
                "extra_data": {
                    "event": "rag_qa_success",
                    "query": query,
                    "source_count": len(sources)
                }
            })
            
            return {
                "response": response.content,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"RAG 问答失败: {e}", exc_info=True, extra={
                "extra_data": {
                    "event": "rag_qa_failed",
                    "query": query
                }
            })
            return {
                "response": "抱歉，系统暂时无法回答您的请求。",
                "sources": []
            }

    async def generate_knowledge_graph(self, doc_id: int, content: str):
        """
        生成知识图谱元数据
        提取关键实体和摘要 (当前为简易版)
        """
        prompt = f"""
        Analyze the following text and extract key concepts (entities) and a brief summary.
        Return the result in JSON format with keys: "summary" (string) and "tags" (list of strings).
        
        Text:
        {content[:2000]} # Limit context window
        """
        try:
            response = await llm.ainvoke(prompt)
            # 简单解析 - 生产环境中应使用 OutputParsers
            content_str = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content_str)
        except Exception as e:
            logger.error(f"生成知识图谱元数据时出错: {e}", exc_info=True, extra={
                "extra_data": {
                    "event": "knowledge_graph_failed",
                    "document_id": doc_id
                }
            })
            return {"summary": "分析失败", "tags": []}

