"""
=============================================================================
文件: agent.py
描述: AI 智能体 API 接口层

核心功能：
1. 对话接口：提供 Chat、RAG 问答的统一入口。
2. 内容辅助：提供润色 (Polish)、续写 (Complete) 等写作辅助功能。

架构职责：
1. 请求解析：使用 Pydantic 模型校验前端传来的 JSON 数据。
2. 依赖注入：获取 AgentService 实例。
3. 逻辑委托：将具体业务逻辑（如调用 OpenAI、查向量库）全权交给 Service 层处理。

依赖组件:
- AgentService: 核心 AI 业务逻辑提供者。
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
    """
    依赖注入: 获取 Agent 服务实例
    
    Why:
    - 确保每个请求都有独立的 DB Session。
    - 方便单元测试时 Mock Service。
    """
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
    与 AI 助手聊天接口
    
    Logic Flow:
        1. **提取问题**: 从请求体中拿出用户最后一条消息。
        2. **模式分发**: 
           - 如果 `use_rag=True`，调用 `rag_qa` (检索增强生成)。
           - 如果 `use_rag=False`，调用 `ask_ai` (纯闲聊)。
        3. **格式封装**: 将结果统一封装成 `ChatResponse` 返回给前端。
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
    内容润色接口
    
    Input: {"content": "写得很烂的原始文本"}
    Output: {"result": "润色后的优美文本"}
    
    Logic Flow:
        直接调用 Service 层的 `polish_text` 方法，让 AI 帮忙改写。
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
    内容续写接口
    
    Input: {"content": "文章的上半部分..."}
    Output: {"result": "AI 接着写的下半部分..."}
    
    Logic Flow:
        直接调用 Service 层的 `complete_text` 方法，让 AI 根据上文猜下文。
    """
    content = text.get("content", "")
    completed = await agent_service.complete_text(content)
    return {"result": completed}
