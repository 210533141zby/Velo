from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import complete_code

router = APIRouter()

class CompletionRequest(BaseModel):
    prefix: str
    suffix: str
    language: str = "markdown"

@router.post("/completion")
async def get_completion(req: CompletionRequest):
    # 简单判断：如果前文是空的，就不补全
    if not req.prefix:
        return {"completion": ""}
    
    # 根据语言动态调整 Max Tokens (代码行短，文本长)
    max_tokens = 64 if req.language == "python" else 32

    content = await complete_code(req.prefix, req.suffix, max_tokens)
    return {"completion": content}