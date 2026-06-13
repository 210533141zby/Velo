from __future__ import annotations

import re

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import ChatMessage, ChatRequest, ChatResponse
from app.services.rag.rag_service import RagService

rag_router = APIRouter(prefix='/agent', tags=['agent'])

FOLLOW_UP_MARKERS = (
    '另一',
    '另一个',
    '另一株',
    '另外',
    '那另一个',
    '那另一',
    '还有',
    '还',
    '然后呢',
    '接着呢',
    '继续',
    '再一个',
    '第二个',
    '第二株',
    '其余',
    '剩下',
)


def _collapse_spaces(value: str) -> str:
    """把消息文本压成单行，便于后续规则判断和拼接查询。

    多轮消息里经常会混入换行、重复空格和复制粘贴残留，本函数先做
    最基础的清洗，避免追问识别被格式噪声干扰。
    """
    return ' '.join(str(value or '').split()).strip()


def _clean_message_text(value: str) -> str:
    """去掉消息首尾的空白和弱语气标点，保留核心语义内容。"""
    return _collapse_spaces(value).strip('，,。.!！？?；;：: ')


def _last_user_message(messages: list[ChatMessage]) -> tuple[int, str]:
    """从会话消息末尾回溯，定位最近一次用户提问。

    返回值同时带上索引位置，便于继续向前查找上一轮用户问题和助手回答，
    用来处理“另一株呢”这类省略主体的追问。
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == 'user':
            return index, _collapse_spaces(message.content)
    return -1, ''


def _is_follow_up_query(query: str) -> bool:
    """根据长度和提示词判断当前问题是否像追问而非独立提问。

    这一步只做轻量启发式识别，目标不是语言学上严格判定，而是尽量把
    省略主语、承接上文的短句送入改写逻辑。
    """
    normalized = _collapse_spaces(query)
    if not normalized:
        return False
    if len(normalized) <= 12:
        return True
    if normalized.endswith('呢') and len(normalized) <= 24:
        return True
    return any(marker in normalized for marker in FOLLOW_UP_MARKERS)


def _build_effective_query(messages: list[ChatMessage]) -> str:
    """把多轮对话整理成真正送入 RAG 的检索查询。

    如果当前问题本身已经完整，就直接使用原句；如果像“还有呢”“另一株呢”
    这样的追问，则拼接上一轮问题和回答，补足被省略的主体与上下文。
    """
    latest_index, latest_user_query = _last_user_message(messages)
    if latest_index < 0 or not latest_user_query:
        return ''
    if not _is_follow_up_query(latest_user_query):
        return latest_user_query

    previous_user = ''
    previous_assistant = ''
    for index in range(latest_index - 1, -1, -1):
        message = messages[index]
        content = _collapse_spaces(message.content)
        if not content:
            continue
        if message.role == 'assistant' and not previous_assistant:
            previous_assistant = content
            continue
        if message.role == 'user':
            previous_user = content
            break

    if not previous_user:
        return latest_user_query

    pieces = [_clean_message_text(previous_user)]
    if previous_assistant:
        pieces.append(f'上一轮回答：{_clean_message_text(previous_assistant)}')
    pieces.append(f'补充问题：{_clean_message_text(latest_user_query)}')
    return re.sub(r'[。]{2,}', '。', '。'.join(piece for piece in pieces if piece)).strip('。')


def get_rag_service(db: AsyncSession = Depends(get_db)) -> RagService:
    """为当前请求创建 RAG 服务实例。"""
    return RagService(db)


@rag_router.post('/chat', response_model=ChatResponse)
async def chat_with_rag(
    request: ChatRequest,
    rag_service: RagService = Depends(get_rag_service),
):
    """整理会话中的有效查询后，调用知识库问答主链路。

    路由层只做对话改写与结果封装，不直接介入检索、重排和作答细节，

    """
    user_query = _build_effective_query(request.messages)
    result = await rag_service.rag_qa(user_query)
    return ChatResponse(
        response=result.get('response', ''),
        sources=result.get('sources'),
    )
