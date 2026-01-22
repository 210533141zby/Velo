"""
=============================================================================
文件: llm_service.py
描述: LLM 底层服务调用 (重构版 - 针对中文幽灵文本优化)

核心功能：
1. 幽灵文本补全：封装对 vLLM 的调用，专为中文编辑器体验优化。
2. FIM 格式构建：使用 Qwen2.5 的 FIM 格式。
3. 上下文裁剪：精细控制上下文长度，降低延迟。

依赖组件:
- httpx: 用于发送异步 HTTP 请求。
- app.core.config: 获取 vLLM 地址配置。
=============================================================================
"""

import httpx
import re
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

def _truncate_context(prefix: str, suffix: str) -> tuple[str, str]:
    """
    上下文精细裁剪 (降低延迟的核心)
    
    Why:
        减少 V100 显存占用，降低首字延迟 (TTFT)。
    
    Args:
        prefix: 光标前的文本
        suffix: 光标后的文本
        
    Returns:
        (truncated_prefix, truncated_suffix)
    """
    # Prefix (前文)：仅保留光标前最后 1000 个字符
    # 约 500 汉字，足够判断语境
    truncated_prefix = prefix[-1000:]
    
    # Suffix (后文)：仅保留光标后最前 200 个字符
    # 用于 FIM 衔接，防止重复
    truncated_suffix = suffix[:200]
    
    return truncated_prefix, truncated_suffix

def _remove_prefix_overlap(text: str, prefix: str) -> str:
    """
    去除生成文本开头与 prefix 结尾的重复部分
    """
    if not text or not prefix:
        return text
        
    # 检查最大可能的重叠长度
    # 我们只需要检查 text 的前缀是否出现在 prefix 的后缀中
    # 从 text 的最大长度开始尝试匹配
    max_overlap = min(len(text), len(prefix))
    
    for i in range(max_overlap, 0, -1):
        # text 的前 i 个字符
        candidate = text[:i]
        # prefix 的后 i 个字符
        target = prefix[-i:]
        
        if candidate == target:
            return text[i:]
            
    return text

async def complete_text(prefix: str, suffix: str) -> str:
    """
    调用 vLLM 服务进行幽灵文本补全 (Ghost Text)
    
    Logic Flow:
        1. 裁剪上下文。
        2. 构造 Qwen FIM Prompt。
        3. 发送请求到 vLLM。
        4. 后处理：去重、空值检查。
    """
    logger.info(f"[GhostBackend] 收到请求 - Prefix长度: {len(prefix)}, Suffix长度: {len(suffix)}")

    # 1. 上下文裁剪
    safe_prefix, safe_suffix = _truncate_context(prefix, suffix)
    
    # 2. 改用 Instruct 指令格式，显式告知前后文
    prompt = (
        "<|im_start|>system\n"
        "你是一个文本补全工具。根据【前文】和【后文】补全中间内容。只输出补全的文字，不要输出任何解释，不要重复前后文。\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"【前文】：...{safe_prefix[-500:]}\n" # 只取最近500字避免混乱
        f"【后文】：{safe_suffix[:200]}...\n"
        "请直接输出补全内容：\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    logger.info(f"[GhostBackend] 向 vLLM 发送 Prompt (长度 {len(prompt)})")

    # 3. 构造请求载荷
    payload = {
        # 必须与 docker-compose 里的模型名一致
        "model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        "prompt": prompt,
        "max_tokens": 24,
        "temperature": 0.1,
        "repetition_penalty": 1.05,
        # 【关键修改】移除 \n 和标点，防止模型起手被杀
        # 让模型先生成，我们在下面代码里截断
        "stop": ["<|endoftext|>", "<|file_sep|>", "<|im_end|>"],
        "stream": False
    }
    
    # 确定 API 地址
    # 优先读取配置，否则回退到 Docker 内部地址
    base_url = settings.VLLM_API_URL
    if not base_url:
        url = "http://vllm:8000/v1/completions"
    else:
        # 兼容处理：如果配置的是 base url (如 http://localhost:8001/v1)，则追加 /completions
        url = f"{base_url.rstrip('/')}/completions"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                timeout=5.0 # 5秒超时，补全必须快
            )
            response.raise_for_status()
            result = response.json()
            
            # 提取结果
            generated_text = ""
            # 适配 vLLM 不同的返回格式 (兼容 choice/content)
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["text"]
            else:
                generated_text = result.get("content", "")
            
            # 4. 后处理 logic

            # 【关键修改】后处理逻辑：外科手术式截断

            # 1. 去除首部可能的换行或空格（Coder模型有时候喜欢先换行）
            generated_text = generated_text.lstrip()

            # 2. 智能切分：遇到 逗号、句号、感叹号、问号、换行 就停止
            # 这样用户永远只得到一个“短语”，不会有负担
            split_text = re.split(r'[，,。！!？?;；\n]', generated_text)
            
            # 3. 取第一段
            final_text = split_text[0]
            
            return final_text

    except httpx.RequestError as e:
        logger.error(f"vLLM 请求失败: {e}")
        # 补全失败不应中断主流程，返回空字符串即可
        return ""
    except Exception as e:
        logger.exception(f"[GhostBackend] 代码补全服务异常: {e}")
        return ""
