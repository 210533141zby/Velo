import axios from 'axios';
import { ElMessage } from 'element-plus';

const service = axios.create({
  baseURL: '/api/v1', // Proxy will handle /api -> http://localhost:8000
  timeout: 10000,
});

service.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API Error:', error);
    ElMessage.error(error.message || '请求失败');
    return Promise.reject(error);
  }
);

export default service;
