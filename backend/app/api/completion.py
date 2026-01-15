"""
=============================================================================
文件: completion.py
描述: 代码补全接口

核心功能：
1. 代码智能提示：根据用户当前光标位置的前文 (prefix) 和后文 (suffix)，预测中间可能输入的代码。

依赖组件:
- llm_service: 提供底层的大模型调用能力。
=============================================================================
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import complete_code

router = APIRouter()

class CompletionRequest(BaseModel):
    """
    代码补全请求参数
    
    Logic Flow:
        FIM (Fill-In-the-Middle) 模式的标准参数。
        - prefix: 光标前面的代码
        - suffix: 光标后面的代码
        - language: 当前文件的语言 (python, markdown, js 等)
    """
    prefix: str
    suffix: str
    language: str = "markdown"

@router.post("/completion")
async def get_completion(req: CompletionRequest):
    """
    获取代码补全建议
    
    Logic Flow:
        1. **参数校验**: 如果前文为空，没法补全，直接返回空串。
        2. **参数调整**: 根据语言类型调整 max_tokens。
           - Python 代码通常逻辑密度高，生成的行数少，给 64 tokens。
           - Markdown 文本可能比较长，但为了响应速度，这里暂时给 32 tokens。
        3. **模型调用**: 委托给 `llm_service.complete_code` 去调大模型。
    """
    # 简单判断：如果前文是空的，就不补全
    if not req.prefix:
        return {"completion": ""}
    
    # 根据语言动态调整 Max Tokens (代码行短，文本长)
    max_tokens = 64 if req.language == "python" else 32

    content = await complete_code(req.prefix, req.suffix, max_tokens)
    return {"completion": content}
