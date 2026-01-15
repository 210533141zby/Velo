"""
=============================================================================
文件: agent.py
描述: AI 智能体 API 接口层
      本模块作为 AgentService 的 HTTP 访问入口，遵循 RESTful 规范。
      
架构准则 (API Layer Responsibilities):
1. **Request Parsing**: 使用 Pydantic 模型 (ChatRequest) 解析和校验请求体。
2. **Dependency Injection**: 通过 `Depends(get_agent_service)` 获取 Service 实例。
3. **Delegation**: 将业务逻辑完全委托给 `AgentService`，API 层不包含复杂的 if/else 业务判断。
4. **Response Formatting**: 将 Service 返回的结果封装为标准 JSON 响应。

依赖关系:
- AgentService: 核心业务逻辑提供者。
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
    1. **Extract Query**: 从请求体中提取用户最新的消息内容。
    2. **Dispatch**: 根据 `request.use_rag` 标志位，决定调用 RAG 模式还是纯聊天模式。
       - True -> `agent_service.rag_qa()`: 检索 + 生成。
       - False -> `agent_service.ask_ai()`: 直接生成。
    3. **Response**: 构造 `ChatResponse` 对象返回。
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
    
    Input: {"content": "原始文本"}
    Output: {"result": "润色后的文本"}
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
    
    Input: {"content": "上下文文本"}
    Output: {"result": "续写的文本"}
    """
    content = text.get("content", "")
    completed = await agent_service.complete_text(content)
    return {"result": completed}
