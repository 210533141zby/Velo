import os
import asyncio
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

# =============================================================================
# 全局常量与配置
# =============================================================================

# 向量存储路径
# 确保路径指向 backend/chroma_db
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

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
        使用 Markdown 专用切分器以优化结构化信息的保留
        """
        if not content:
            return

        try:
            # 1. 使用 MarkdownHeaderTextSplitter 切分
            # 定义要识别的 Markdown 标题层级
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(content)

            # 2. 使用 RecursiveCharacterTextSplitter 进一步切分
            # 即使有了 header split，某些段落可能仍然过长，或者没有 header
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_splits = text_splitter.split_documents(md_header_splits)
            
            # 如果 Markdown 切分没有产生结果（例如纯文本无标题），回退到直接切分
            if not final_splits and content:
                final_splits = text_splitter.create_documents([content])

            # 3. 构造 LangChain Documents 并添加元数据
            docs = []
            for split in final_splits:
                # 确保 metadata 存在
                if not split.metadata:
                    split.metadata = {}
                
                # 添加来源和 ID
                split.metadata["source"] = title
                split.metadata["doc_id"] = doc_id
                docs.append(split)
            
            if not docs:
                return

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
        # 实际实现可能需要更复杂的解析，这里仅占位
        pass
