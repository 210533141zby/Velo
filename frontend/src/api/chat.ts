import client from './client';
import type { ChatResponsePayload } from '@/types/chat';

const CHAT_REQUEST_TIMEOUT_MS = 120000;

/**
 * 发送一轮知识库问答请求。
 */
export function sendChatMessage(content: string) {
  return client.post<never, ChatResponsePayload>(
    '/agent/chat',
    {
      messages: [{ role: 'user', content }],
    },
    {
      timeout: CHAT_REQUEST_TIMEOUT_MS,
    },
  );
}
