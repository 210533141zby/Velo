import axios, { AxiosError } from 'axios';
import { ElMessage } from 'element-plus';

type ApiErrorPayload = {
  detail?: string | { message?: string } | Array<{ msg?: string }>;
  message?: string;
};

/**
 * 把 axios 错误整理成适合前端提示的文案。
 */
function resolveErrorMessage(error: AxiosError<ApiErrorPayload>) {
  const payload = error.response?.data;

  if (!error.response) {
    return '无法连接后端服务，请确认 127.0.0.1:8000 已启动';
  }

  if (error.code === 'ECONNABORTED' || error.message.toLowerCase().includes('timeout')) {
    return '请求超时，请稍后重试';
  }

  if (typeof payload?.detail === 'string') {
    return payload.detail;
  }

  if (Array.isArray(payload?.detail) && payload.detail[0]?.msg) {
    return payload.detail[0].msg;
  }

  if (payload?.message) {
    return payload.message;
  }

  return error.message || '请求失败';
}

/**
 * 获取请求error消息。
 */
export function getRequestErrorMessage(error: unknown) {
  if (axios.isAxiosError<ApiErrorPayload>(error)) {
    return resolveErrorMessage(error);
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return '请求失败';
}

const client = axios.create({
  baseURL: '/api/v1',
  timeout: 15000,
});

client.interceptors.response.use(
  (response) => response.data,
  (error: AxiosError<ApiErrorPayload>) => {
    if (error.code !== 'ERR_CANCELED') {
      console.error('API Error:', error);
      ElMessage.error(resolveErrorMessage(error));
    }

    return Promise.reject(error);
  },
);

export default client;
