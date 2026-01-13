import httpx
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

async def complete_code(prefix: str, suffix: str, max_tokens: int = 512) -> str:
    """
    Call vLLM service to complete code using FIM (Fill-In-the-Middle).
    
    Args:
        prefix: The code before the cursor.
        suffix: The code after the cursor.
        max_tokens: Maximum tokens to generate.
        
    Returns:
        The generated code completion.
    """
    # Construct FIM Prompt for Qwen2.5-Coder
    # Format: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>
    prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    
    # Parameters for Qwen2.5-Coder
    payload = {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct", # Adjust model name as per your vLLM deployment
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.95,
        "stop": ["<|file_separator|>", "\n\n"], # Stop on file separator or double newline
        "stream": False
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
            
            # Extract completion text
            # vLLM returns standard OpenAI completion format
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"]
            else:
                return ""
                
    except httpx.RequestError as e:
        logger.error(f"An error occurred while requesting vLLM: {e}")
        raise e
    except httpx.HTTPStatusError as e:
        logger.error(f"Error response {e.response.status_code} while requesting vLLM: {e.response.text}")
        raise e
