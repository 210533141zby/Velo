export interface CompletionRequestPayload {
  prefix: string;
  suffix: string;
  language: string;
  trigger_mode: 'manual' | 'auto';
}

export interface CompletionResponsePayload {
  completion: string;
  reason?: string | null;
}

/**
 * 探测补全服务是否可用。
 */
export async function checkCompletionBackendHealth(signal?: AbortSignal) {
  try {
    const response = await fetch('/health', {
      method: 'GET',
      cache: 'no-store',
      signal,
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * 请求一次文本补全，并规范化返回结构。
 */
export async function requestCompletion(payload: CompletionRequestPayload, signal: AbortSignal) {
  const response = await fetch('/api/v1/completion', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Completion request failed: ${response.status}`);
  }

  const result = (await response.json()) as CompletionResponsePayload;
  return {
    completion: result.completion ?? '',
    reason: result.reason ?? null,
  };
}
