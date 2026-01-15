"""
=============================================================================
文件: llm_service.py
描述: LLM 底层服务调用

核心功能：
1. 代码补全：封装对 vLLM (兼容 OpenAI 接口) 的调用。
2. FIM 格式构建：为代码补全模型 (如 Qwen2.5-Coder) 构建 Fill-In-the-Middle 提示词。

依赖组件:
- httpx: 用于发送异步 HTTP 请求。
- app.core.config: 获取 vLLM 地址配置。
=============================================================================
"""

import httpx
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

async def complete_code(prefix: str, suffix: str, max_tokens: int = 512) -> str:
    """
    调用 vLLM 服务进行代码补全 (FIM 模式)
    
    Logic Flow:
        1. **构建提示词**: 使用 Qwen2.5-Coder 的 FIM (Fill-In-the-Middle) 格式。
           格式: `<|fim_prefix|>{前文}<|fim_suffix|>{后文}<|fim_middle|>`
           这样模型就知道它需要生成中间的代码。
        2. **准备参数**: 
           - `stop`: 设置停止词，防止模型生成多余的文件分隔符。
           - `stream`: 设为 False，因为这里我们是等待完整响应再返回，而不是流式输出。
        3. **发送请求**: 异步调用 vLLM 的 `/completions` 接口。
        4. **结果提取**: 从 OpenAI 兼容的响应格式中提取 `text` 字段。
        
    Why:
        为什么 `stream` 设为 `False`？
        代码里使用了 `response.json()` 解析响应，这要求服务器返回完整的 JSON 包。
        如果开启流式 (`stream=True`)，服务器会返回 SSE (Server-Sent Events) 数据流，
        直接调 `.json()` 会报错或解析失败。
    """
    # 构造 Qwen2.5-Coder 的 FIM Prompt
    prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    
    # 构造请求载荷
    payload = {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct", # 根据实际部署的模型名称调整
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1, # 低温度，让代码生成更稳定
        "top_p": 0.95,
        "stop": ["<|file_separator|>", "\n\n"], # 遇到文件分隔符或双换行就停止
        "stream": False # 必须关闭流式，否则下面 response.json() 无法解析
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.VLLM_API_URL}/completions",
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            
            # 提取补全结果
            # vLLM 返回标准的 OpenAI Completion 格式
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"]
            else:
                return ""
                
    except httpx.RequestError as e:
        logger.error(f"请求 vLLM 服务发生错误: {e}")
        raise e
    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM 服务返回错误状态码 {e.response.status_code}: {e.response.text}")
        raise e
