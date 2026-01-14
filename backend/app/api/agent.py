"""
=============================================================================
文件: agent.py
描述: AI 智能体接口
      提供与 AI 助手对话、文本润色、内容补全等功能的 RESTful API
      核心逻辑封装在 AgentService 中
=============================================================================
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import ChatRequest, ChatResponse
from app.services.agent_service import AgentService

router = APIRouter()

# -----------------------------------------------------------------------------
# 辅助函数: 获取服务实例
# -----------------------------------------------------------------------------

def get_agent_service(db: AsyncSession = Depends(get_db)) -> AgentService:
    """依赖注入: 获取 Agent 服务实例"""
    return AgentService(db)

# -----------------------------------------------------------------------------
# 路由定义
# -----------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    与 AI 助手聊天
    
    - 支持普通对话模式
    - 支持 RAG (检索增强生成) 模式，基于本地知识库回答
    """
    user_query = request.messages[-1].content
    
    # 根据请求参数决定是否使用 RAG
    if request.use_rag:
        result = await agent_service.rag_qa(user_query)
        return ChatResponse(
            response=result.get("response", ""),
            sources=result.get("sources")
        )
    else:
        # 纯 AI 对话
        response_text = await agent_service.ask_ai(user_query)
        return ChatResponse(
            response=response_text,
            sources=None
        )

@router.post("/polish")
async def polish_content(
    text: dict, 
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    内容润色
    
    - 调用 LLM 对文本进行润色，提升专业度
    """
    content = text.get("content", "")
    polished = await agent_service.polish_text(content)
    return {"result": polished}

@router.post("/complete")
async def complete_content(
    text: dict, 
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    内容续写
    
    - 调用 LLM 根据上下文自动补全内容
    """
    content = text.get("content", "")
    completed = await agent_service.complete_text(content)
    return {"result": completed}
